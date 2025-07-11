import cv2
import numpy as np
import os
from utils.data_loader import DatasetAdv, ApplyTexture
import neural_renderer
from torch.utils.data import DataLoader
import torch
from train import collate_fn
import tqdm
from itertools import islice

# -----------------------------
# 显著性图生成函数（可选 OpenCV 方法）
# -----------------------------
def compute_saliency_opencv(image, method='spectral'):
    """
    使用 OpenCV 提供的方法生成显著性图
    :param image: 输入图像 (BGR)
    :param method: 'spectral' 或 'fine_grained'
    :return: 显著性图 (灰度图)
    """
    if method == 'spectral':
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, saliency_map = saliency.computeSaliency(image)
        # saliency_map = (saliency_map * 255).astype("uint8")
    elif method == 'fine_grained':
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        _, saliency_map = saliency.computeSaliency(image)
    else:
        raise ValueError("method must be 'spectral' or 'fine_grained'")
    return saliency_map


# -----------------------------
# YOLO 标签解析
# -----------------------------
def get_yolo_boxes(label_path, img_shape):
    """
    解析 YOLO 格式标签文件
    :param label_path: 标签路径
    :param img_shape: 图像形状 (h, w, c)
    :return: list of (class_id, x1, y1, x2, y2)
    """
    boxes = []
    if not os.path.exists(label_path):
        print("没有找到标签文件")
        return boxes

    with open(label_path, 'r') as f:
        lines = f.readlines()

    img_h, img_w = img_shape[:2]
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        x = int((cx - w / 2) * img_w)
        y = int((cy - h / 2) * img_h)
        w_px = int(w * img_w)
        h_px = int(h * img_h)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w_px), min(img_h, y + h_px)
        boxes.append((class_id, x1, y1, x2, y2))
    return boxes


# -----------------------------
# 计算显著性得分
# -----------------------------
def get_saliency_scores(image, label_path, method):
    """
    计算每个目标区域的显著性得分
    :param image: 输入图像 (BGR)
    :param label_path: YOLO 标签路径
    :return: 每个目标的显著性得分列表
    """
    saliency_map = compute_saliency_opencv(image, method=method)
    boxes = get_yolo_boxes(label_path, image.shape)

    scores = []
    for class_id, x1, y1, x2, y2 in boxes:
        roi = saliency_map[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        mean_score = roi.mean()
        max_score = roi.max()
        high_ratio = np.mean(roi > 200)
        scores.append({
            "class_id": class_id,
            "mean_score": mean_score,
            "max_score": max_score,
            "high_ratio": high_ratio
        })
    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imwrite("test.png", image)
    return scores


# -----------------------------
# 支持 Tensor 输入
# -----------------------------
def get_saliency_scores_from_tensor(tensor, label_path, method):
    """
    接收 PyTorch 张量格式图像，转换为 OpenCV 格式后进行显著性分析
    :param tensor: shape=(C, H, W)
    :param label_path: YOLO 标签路径
    :return: 每个目标的显著性得分列表
    """
    # 转换为 NumPy 数组
    img = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return get_saliency_scores(img, label_path, method)


# -----------------------------
# 热力图可视化
# -----------------------------
def visualize_saliency(image, saliency_map, boxes=None, save_dir=None, show=False):
    """
    显著性热力图可视化，并标注目标边界框
    :param image: 原始图像
    :param saliency_map: 显著性图
    :param boxes: list of (class_id, x1, y1, x2, y2)
    :param save_dir: 保存目录
    :param show: 是否显示图像
    """
    heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.3, heatmap, 0.7, 0)

    if boxes:
        for class_id, x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "original_with_bbox.png"), image)
        cv2.imwrite(os.path.join(save_dir, "heatmap.png"), heatmap)
        cv2.imwrite(os.path.join(save_dir, "overlay.png"), overlay)

    if show:
        cv2.imshow("Original", image)
        cv2.imshow("Heatmap", heatmap)
        cv2.imshow("Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def vis_sal(image_path, label_path, save_dir="output", show=True):
    # 步骤 1：读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    # 步骤 2：计算显著性图
    saliency_map = compute_saliency_opencv(image, method='fine_grained')

    # 步骤 3：读取 YOLO 标签并解析为目标框
    boxes = get_yolo_boxes(label_path, image.shape)

    # 步骤 4：调用可视化函数
    visualize_saliency(image=image,
                       saliency_map=saliency_map,
                       boxes=boxes,
                       save_dir=save_dir,
                       show=show)

    print(f"可视化完成，结果保存在: {save_dir}")


def cal_sal(textures_file, method):
    # load 3d model
    texture_size = 6
    image_size = 800
    obj_file = '/root/ovd-attack/data/test.obj'
    vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, texture_size=texture_size,
                                                            load_texture=True)
    # Camouflage Textures
    faces_file = '/root/ovd-attack/data/faces_new.txt'
    texture_content_adv = torch.from_numpy(
        np.load(textures_file)).cuda(device=0)
    texture_origin = textures[None, :, :, :, :, :].cuda(device=0)
    # test
    texture_mask = np.zeros((faces.shape[0], 6, 6, 6, 3), 'int8')
    with open(faces_file, 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            if face_id != '\n':
                texture_mask[int(face_id) - 1, :, :, :, :] = 1
    texture_mask = torch.from_numpy(
        texture_mask).cuda(device=0).unsqueeze(0)
    mask_dir = '/root/autodl-tmp/dataset/val/mask/'
    data_dir = '/root/autodl-tmp/dataset/val/npz/'
    dataset = DatasetAdv(
        data_dir, 'None', image_size, texture_size, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=True)

    def cal_texture(texture_content):
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
        return texture_origin * (1 - texture_mask) + texture_mask * textures
    
    def cal_texture_(texture_content):
        textures = texture_content.clamp(0, 1)
        return texture_origin * (1 - texture_mask) + texture_mask * textures

    def cal_texture__(texture_content):
        min_color = torch.tensor(
            [0.03, 0.08, 0.02], device=texture_content.device)
        max_color = torch.tensor(
            [0.25, 0.45, 0.18], device=texture_content.device)
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
        textures = textures * (max_color - min_color) + min_color
        textures = torch.clamp(textures, 0.0, 1.0)
        return texture_origin * (1 - texture_mask) + texture_mask * textures

    if 'textures_iou_clamp' in textures_file:
        cal_texture_fun = cal_texture__
    elif 'textures_sp' in textures_file:
        cal_texture_fun = cal_texture
    else:
        cal_texture_fun = cal_texture_

    textures_adv = cal_texture_fun(texture_content_adv)
    dataset.set_textures(textures_adv)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    sample_num = len(data_loader)
    # pbar = enumerate(data_loader)
    # pbar = tqdm.tqdm(pbar, total=sample_num)

    slice = 500
    pbar = islice(enumerate(data_loader), slice)
    pbar = tqdm.tqdm(pbar, total=slice)
    total_score, count = 0.0, 0
    for i, sample in pbar:
        name = sample[0]['filename']
        label_path = f'/root/autodl-tmp/dataset/val/label/{name.split(".")[0]}.txt'
        scores_list = get_saliency_scores_from_tensor(sample[0]['data']['inputs'][0], label_path, method)
        for score_dict in scores_list:
            total_score += score_dict['mean_score']
            count += 1

    avg_score = round(total_score / count, 4) if count > 0 else 0.0
    filename = os.path.basename(textures_file).replace(".npy", ".txt")
    output_dir = f"saliency_{method}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        f.write(f"{avg_score}\n")
    print(f"已保存显著性得分到: {output_path}")

def calculate_average_saliency_score(folder_path, method='fine_grained'):
    """
    计算文件夹中所有 .png 图像的平均显著性分数
    :param folder_path: 包含图像的文件夹路径
    :param method: 显著性检测方法 ('spectral' 或 'fine_grained')
    :return: 平均显著性分数
    """
    def compute_saliency_opencv(image, method='fine_grained'):
        """
        使用 OpenCV 提供的方法生成显著性图
        :param image: 输入图像 (BGR)
        :param method: 'spectral' 或 'fine_grained'
        :return: 显著性图 (uint8, 0~255)
        """
        if method == 'spectral':
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            _, saliency_map = saliency.computeSaliency(image)
            saliency_map = (saliency_map * 255).astype("uint8")
        elif method == 'fine_grained':
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            _, saliency_map = saliency.computeSaliency(image)
            saliency_map = (saliency_map * 255).astype("uint8")
        else:
            raise ValueError("method must be 'spectral' or 'fine_grained'")
        return saliency_map
    # 获取文件夹中的所有 .png 文件
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # 存储所有图像的显著性分数
    mean_scores = []
    
    for file_name in png_files:
        # 构建完整路径
        file_path = os.path.join(folder_path, file_name)
        
        # 读取图像
        image = cv2.imread(file_path)
        if image is None:
            print(f"无法加载图像: {file_path}")
            continue
        
        # 计算显著性图
        saliency_map = compute_saliency_opencv(image, method=method)
        
        # 计算显著性分数（平均值）
        mean_score = saliency_map.mean()
        mean_scores.append(mean_score)
    
    # 计算平均显著性分数
    avg_mean_score = np.mean(mean_scores) if mean_scores else 0
    
    return avg_mean_score


if __name__ == "__main__":

    # 整个验证集
    # texture_list = [
    #     'textures_iou_clamp/dino/texture_camouflage_dino_multi_epoch_5.npy',
    #     'textures_sp/dino/texture_camouflage_dino_single_epoch_5.npy',
    #     'textures_iou_clamp/glip/texture_camouflage_glip_multi_epoch_5.npy',
    #     'textures_sp/glip/texture_camouflage_glip_single_epoch_5.npy',
    #     'textures_iou_clamp/yolo/texture_camouflage_yolo_multi_epoch_5.npy',
    #     'textures_sp/yolo/texture_camouflage_yolo_single_epoch_5.npy',
    # ]
    # method_list = ['spectral', 'fine_grained']
    # for method in method_list:
    #     for textures_file in texture_list:
    #         cal_sal(textures_file, method)

    # 渲染的图像的显著性
    # spectral
    # \multirow{2}{*}{GroundingDino} & SP-Attack & 25.07 & 11.46 & 5.73 & 4.05 & \\
    #                                & MP-Attack & 19.97 & 10.07 & 6.75 & 5.11 & \\
    # \hline
    # \multirow{2}{*}{GLIP} & SP-Attack & 23.53 & 10.43 & 6.07 & 4.38 & \\
    #                       & MP-Attack & 22.02 & 10.67 & 5.79 & 4.58 & \\
    # \hline
    # \multirow{2}{*}{YOLO-World} & SP-Attack & 22.54 & 11.78 & 6.35 & 4.19 & \\
    #                             & MP-Attack & 22.04 & 11.58 & 6.70 & 4.85 & \\ 
    # fine_grained
    # \multirow{2}{*}{GroundingDino} & SP-Attack & 61.47 & 29.92 & 19.91 & 15.52 & \\
    #                                & MP-Attack & 51.07 & 27.66 & 19.60 & 15.02 & \\
    # \hline
    # \multirow{2}{*}{GLIP} & SP-Attack & 58.54 & 29.46 & 20.45 & 15.82 & \\
    #                       & MP-Attack & 59.37 & 29.91 & 20.10 & 14.84 & \\
    # \hline
    # \multirow{2}{*}{YOLO-World} & SP-Attack & 65.66 & 31.61 & 20.85 & 15.98 & \\
    #                             & MP-Attack & 59.47 & 30.17 & 20.60 & 15.22 & \\    
    for folder_path in ["results_iou/dis_2", "results_iou/dis_4"]:
        method = 'fine_grained'  # 或者 fine_grained spectral
        avg_score = calculate_average_saliency_score(folder_path, method)
        print(f"所有图像的平均显著性分数: {avg_score:.2f}")

