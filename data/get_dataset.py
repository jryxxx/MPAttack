import glob
import os
import sys
import math
import numpy as np
import cv2
import carla # type: ignore
from queue import Queue, Empty

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

### 需要修改的地方
# 对于装甲车，使用 Town06，选择 cougar，颜色设置为 0，0，0
# 对于民用车辆，使用 Town01，选择 etron_my，颜色设置为 142，0，0

# ==================== 用户配置参数 ====================
DATASET_DIR = "dataset"     # 数据集输出路径
NUM_POSITIONS = 200          # 每种天气下采样的车辆位置数
VIEWPOINTS = {
    "azimuth": list(range(0, 360, 30)),       # 方位角：每 30° 一次
    "elevation": [-45, -60, -75, -90],       # 高度角
    "distance": [30]                         # 距离
}
IMAGE_SIZE_X = '800'
IMAGE_SIZE_Y = '800'


# ==================== 工具函数 ====================
def get_relative_camera_transform(azimuth_deg, elevation_deg, distance):
    rad_azimuth = math.radians(azimuth_deg)
    rad_elevation = math.radians(elevation_deg)

    x = distance * math.cos(rad_elevation) * math.cos(rad_azimuth)
    y = distance * math.cos(rad_elevation) * math.sin(rad_azimuth)
    z = distance * math.sin(rad_elevation)
    x, y, z = -x, -y, -z
    yaw = azimuth_deg
    pitch = elevation_deg
    roll = 0
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
    ), x, y, z


def generate_weather_combinations():
    return [{'fog_density': 0, 'sun_altitude_angle': 30, 'precipitation_deposits': 0}]


def get_carla_transform(transform):
    loc = transform.location
    rot = transform.rotation
    return [[loc.x, loc.y, loc.z], [rot.pitch, rot.yaw, rot.roll]]


def save_image(rgb_data=None, seg_data=None, vehicle=None, camera_transform=None,
               weather=None, position_idx=0, view_idx=0, az=0, el=0, dist=0,
               x=0, y=0, z=0, output_dir="dataset"):
    npz_folder = os.path.join(output_dir, "npz")
    mask_folder = os.path.join(output_dir, "mask")

    os.makedirs(npz_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    az_str = f"{az:.1f}".replace('.', '_')
    el_str = f"{el:.1f}".replace('.', '_')
    dist_str = f"{dist:.1f}".replace('.', '_')
    filename_base = f"pos_{position_idx}_az{az_str}_el{el_str}_dist{dist_str}"
    filename_npz = filename_base + ".npz"
    filename_mask = filename_base + ".png"

    print(f"Saving: {filename_npz if rgb_data is not None else ''}, {filename_mask if seg_data is not None else ''}")

    # 保存 RGB npz
    if rgb_data is not None:
        array_rgb = np.frombuffer(rgb_data.raw_data, dtype=np.uint8).reshape((rgb_data.height, rgb_data.width, 4))[:, :, :3]
        veh_trans = get_carla_transform(vehicle.get_transform())
        cam_trans = get_carla_transform(camera_transform)

        bbox = vehicle.bounding_box
        cam_trans[0][0] = x - bbox.location.x
        cam_trans[0][1] = y - bbox.location.y
        cam_trans[0][2] = z - bbox.location.z
        cam_trans[1][0] = el
        cam_trans[1][1] = az
        cam_trans[1][2] = 0

        np.savez_compressed(os.path.join(npz_folder, filename_npz), img=array_rgb, veh_trans=veh_trans, cam_trans=cam_trans)

    # 保存 Mask png
    if seg_data is not None:
        seg_data.convert(carla.ColorConverter.CityScapesPalette)
        array_seg = np.frombuffer(seg_data.raw_data, dtype=np.uint8).reshape((seg_data.height, seg_data.width, 4))[:, :, :3]
        # gray_mask = np.all(array_seg == [0, 0, 0], axis=-1) # 装甲车
        gray_mask = np.all(array_seg == [142, 0, 0], axis=-1) # 民用车辆
        binary_mask = np.zeros_like(gray_mask, dtype=np.uint8)
        binary_mask[gray_mask] = 255
        binary_mask_3channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        print(os.path.join(mask_folder, filename_mask))
        cv2.imwrite(os.path.join(mask_folder, filename_mask), binary_mask_3channel)


# ==================== 相机采集函数 ====================
def capture_rgb_sensor(world, vehicle, transform, rgb_bp, timeout=1.0):
    data_queue = Queue()
    camera = world.spawn_actor(rgb_bp, transform, attach_to=vehicle)
    camera.listen(lambda data: data_queue.put(data))

    world.tick()
    try:
        rgb_data = data_queue.get(timeout=timeout)
        return rgb_data, camera
    except Empty:
        print("      RGB image timeout.")
        return None, camera
    finally:
        camera.stop()
        camera.destroy()


def capture_seg_sensor(world, vehicle, transform, seg_bp, timeout=1.0):
    data_queue = Queue()
    camera = world.spawn_actor(seg_bp, transform, attach_to=vehicle)
    camera.listen(lambda data: data_queue.put(data))

    world.tick()
    try:
        seg_data = data_queue.get(timeout=timeout)
        return seg_data, camera
    except Empty:
        print("      Segmentation image timeout.")
        return None, camera
    finally:
        camera.stop()
        camera.destroy()


def collect_data(world, vehicle, transform, cam_rgb_bp, cam_seg_bp, mode='both'):
    """
    Args:
        mode (str): 'rgb', 'mask', or 'both'
    Returns:
        tuple: (rgb_data, seg_data)
    """
    rgb_data, seg_data = None, None

    if mode in ['rgb', 'both']:
        rgb_data, _ = capture_rgb_sensor(world, vehicle, transform, cam_rgb_bp)
    if mode in ['mask', 'both']:
        seg_data, _ = capture_seg_sensor(world, vehicle, transform, cam_seg_bp)

    return rgb_data, seg_data


# ==================== 主函数 ====================
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    client.load_world('Town06') # 装甲车使用 Town06，民用车辆使用 Town01
    world = client.get_world()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = True
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    print(blueprint_library)
    vehicle_bp = blueprint_library.filter('etron')[0] # 装甲车使用 'cougar_color'，民用车辆使用 'etron_my'
    spawn_points = world.get_map().get_spawn_points()
    num_spawn_points = len(spawn_points)
    gap = num_spawn_points // NUM_POSITIONS
    print(f"Avaliable points: {len(spawn_points)}, gap: {gap}")

    cam_rgb_bp = blueprint_library.find('sensor.camera.rgb')
    cam_seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')

    cam_rgb_bp.set_attribute('image_size_x', IMAGE_SIZE_X)
    cam_rgb_bp.set_attribute('image_size_y', IMAGE_SIZE_Y)
    cam_seg_bp.set_attribute('image_size_x', IMAGE_SIZE_X)
    cam_seg_bp.set_attribute('image_size_y', IMAGE_SIZE_Y)

    weathers = generate_weather_combinations()

    try:
        for weather in weathers:
            print(f"\n=== Weather: Fog={weather['fog_density']}, Sun={weather['sun_altitude_angle']}, Pre={weather['precipitation_deposits']} ===")
            world.set_weather(carla.WeatherParameters(**weather))

            viewpoints = [(a, e, d) for a in VIEWPOINTS["azimuth"]
                                    for e in VIEWPOINTS["elevation"]
                                    for d in VIEWPOINTS["distance"]]

            for view_idx, (az, el, dist) in enumerate(viewpoints):
                print(f"  Viewpoint {view_idx + 1}: Azimuth={az}, Elevation={el}, Distance={dist}")

                for pos_idx in range(NUM_POSITIONS):
                    print(f"    === Position {pos_idx + 1}/{NUM_POSITIONS} ===")
                    spawn_point = spawn_points[pos_idx * gap]
                    def set_angle(transform):
                        return carla.Transform(transform.location, carla.Rotation(0, 0, 0))
                    spawn_point = set_angle(spawn_point)
                    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                    if not vehicle:
                        print("      Failed to spawn vehicle.")
                        continue
                    vehicle.set_simulate_physics(False)

                    new_transform, x, y, z = get_relative_camera_transform(az, el, dist)

                    # 控制采集模式：'rgb' / 'mask' / 'both'
                    rgb_data, seg_data = collect_data(world, vehicle, new_transform, cam_rgb_bp, cam_seg_bp, mode='both')

                    # 保存对应格式
                    save_image(rgb_data, seg_data, vehicle, new_transform, weather, pos_idx, view_idx, az, el, dist, x, y, z)

                    vehicle.destroy()

    finally:
        print("\nRestoring original settings...")
        world.apply_settings(original_settings)
        print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')