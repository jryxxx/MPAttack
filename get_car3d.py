import os
import argparse

import torch
import numpy as np
import tqdm
import imageio

import neural_renderer as nr


def rotate_vertices(vertices, axis='x', angle_degrees=90):
    """
    :params vertices: vertices [num_vertices, 3]
    :return vertices: vertices [num_vertices, 3]
    """
    angle = np.radians(angle_degrees)
    if axis == 'x':
        rot = torch.tensor([[1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle),  np.cos(angle)]], dtype=torch.float32)
    elif axis == 'y':
        rot = torch.tensor([[np.cos(angle), 0, np.sin(angle)],
                            [0, 1, 0],
                            [-np.sin(angle), 0, np.cos(angle)]], dtype=torch.float32)
    elif axis == 'z':
        rot = torch.tensor([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle),  np.cos(angle), 0],
                            [0, 0, 1]], dtype=torch.float32)
    else:
        raise ValueError("Invalid axis")

    rot = rot.to(vertices.device)
    return vertices @ rot.T  # [N, 3] × [3, 3]ᵀ = [N, 3]


def get_3d_car():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str,
                        default="data/audi_et_te.obj")
    parser.add_argument('-o', '--filename_output', type=str,
                        default="results/result.gif")
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    camera_distance = 2.732
    elevation = 30

    # load .obj
    texture_size = 2
    vertices, faces = nr.load_obj(
        args.filename_input, texture_size=texture_size)
    print(f"vertices.shape: {vertices.shape}")
    vertices = rotate_vertices(
        vertices, axis='x', angle_degrees=-90)
    print(f"vertices.shape: {vertices.shape}")
    # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    vertices = vertices[None, :, :]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(
        1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    # to gpu

    # create renderer
    renderer = nr.Renderer(camera_mode='look_at')

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    writer = imageio.get_writer(args.filename_output, mode='I')
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = nr.get_points_from_angles(
            camera_distance, elevation, azimuth)
        # [batch_size, RGB, image_size, image_size]
        images, _, _ = renderer(vertices, faces, textures)
        image = images.detach().cpu().numpy()[0].transpose(
            (1, 2, 0))  # [image_size, image_size, RGB]
        writer.append_data((255 * image).astype(np.uint8))
    writer.close()


def texture23d():
    obj_file = 'data/test.obj'
    img_save_dir = 'data/'
    if not (os.path.exists(img_save_dir)):
        os.makedirs(img_save_dir)
    texture_size = 6

    vertices, faces, texture = nr.load_obj(
        obj_file, texture_size=texture_size, load_texture=True)
    texture_origin = texture[None, :, :, :, :, :].cuda(device=0)

    texture_mask = np.zeros(
        (faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
    with open('data/faces_new.txt', 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            texture_mask[int(face_id) - 1, :, :, :, :] = 1
    texture_mask = torch.from_numpy(
        texture_mask).cuda(device=0).unsqueeze(0)
    print(texture_mask.size())
    faces_var = torch.autograd.Variable(faces.cuda(device=0))
    vertices_var = vertices.cuda(device=0)
    textures = np.load(
        'textures_iou_clamp/dino/texture_camouflage_dino_multi_epoch_5.npy')
    textures = torch.from_numpy(textures).cuda(device=0)
    print(textures.size())

    def cal_texture__(texture_content):
        min_color = torch.tensor(
            [0.03, 0.08, 0.02], device=texture_content.device)
        max_color = torch.tensor(
            [0.25, 0.45, 0.18], device=texture_content.device)
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
        textures = textures * (max_color - min_color) + min_color
        textures = torch.clamp(textures, 0.0, 1.0)
        return texture_origin * (1 - texture_mask) + texture_mask * textures
    textures_content = cal_texture__(textures)

    nr.save_obj(img_save_dir + 'final.obj', vertices_var,
                faces_var, textures_content.squeeze(0), texture_size_out=6)


def get_texture_car(dis):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str,
                        default="data/test.obj")
    parser.add_argument('-o', '--filename_output', type=str,
                        default="results_iou_clamp/result.gif")
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    camera_distance = dis
    os.makedirs(f'results_iou/dis_{camera_distance}', exist_ok=True)
    elevation = 60

    # load .obj
    texture_size = 6
    vertices, faces, texture = nr.load_obj(
        args.filename_input, texture_size=texture_size, load_texture=True)
    texture_origin = texture[None, :, :, :, :, :].cuda(device=0)
    texture_mask = np.zeros(
        (faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
    with open('data/faces_new.txt', 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            texture_mask[int(face_id) - 1, :, :, :, :] = 1
    texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)
    texture_content_adv = np.load(
        'textures_iou/dino/texture_camouflage_dino_multi_epoch_5.npy')
    texture_content_adv = torch.from_numpy(texture_content_adv).cuda(device=0)

    # for clamp
    # def cal_texture__(texture_content):
    #     min_color = torch.tensor(
    #         [0.03, 0.08, 0.02], device=texture_content.device)
    #     max_color = torch.tensor(
    #         [0.25, 0.45, 0.18], device=texture_content.device)
    #     textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    #     textures = textures * (max_color - min_color) + min_color
    #     textures = torch.clamp(textures, 0.0, 1.0)
    #     return texture_origin * (1 - texture_mask) + texture_mask * textures

    # for no clamp
    def cal_texture__(texture_content):
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
        return texture_origin * (1 - texture_mask) + texture_mask * textures

    textures = cal_texture__(texture_content_adv)
    vertices = rotate_vertices(
        vertices, axis='x', angle_degrees=-90)
    # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    vertices = vertices[None, :, :]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create renderer
    renderer = nr.Renderer(camera_mode='look_at')

    # draw object
    loop = tqdm.tqdm(range(0, 360, 3))
    writer = imageio.get_writer(args.filename_output, mode='I')
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = nr.get_points_from_angles(
            camera_distance, elevation, azimuth)
        # [batch_size, RGB, image_size, image_size]
        images, _, _ = renderer(vertices, faces, textures)
        image = images.detach().cpu().numpy()[0].transpose(
            (1, 2, 0))  # [image_size, image_size, RGB]
        if num % 15 == 0:
            image_save = f'results_iou/dis_{camera_distance}/texture_{num:03d}_{azimuth}_{elevation}.png'
            imageio.imwrite(image_save, (255 * image).astype(np.uint8))
        writer.append_data((255 * image).astype(np.uint8))
    writer.close()


if __name__ == '__main__':
    # texture23d()

    
    for dis in [2, 4]:
        get_texture_car(dis)
