import torch
import warnings
import cv2
import urllib.request
from pathlib import Path
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet.registry import VISUALIZERS
from mmcv.transforms import Compose
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
import torch.nn.functional as F
from nltk.corpus import wordnet
warnings.filterwarnings("ignore")

coco_class_names = "person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush ."


class GetTrainData:
    """
    input: image(np.ndarray), model(mmdet.models.detectors)
    1. get padding image's shape
    2. get padding image
    3. pack data
    """

    def __init__(self):
        self.device = None

    def is_to_rgb(self, model):
        """check if a model takes rgb images or not
        Args:
            model (~ mmdet.models.detectors): a mmdet model
        """
        self.device = model.device
        to_rgb = model.cfg['model']['data_preprocessor']['bgr_to_rgb']
        return to_rgb

    def prefetch_batch_input_shape(self, model, ori_wh):
        cfg = model.cfg
        w, h = ori_wh
        cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
        test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
        data = {'img': np.zeros((h, w, 3), dtype=np.uint8), 'img_id': 0}
        data = test_pipeline(data)
        data['inputs'] = [data['inputs']]
        data['data_samples'] = [data['data_samples']]
        data_sample = model.data_preprocessor(data, False)['data_samples']
        batch_input_shape = data_sample[0].batch_input_shape
        return batch_input_shape

    def resize_image_keep_ratio(self, image, target_size, align='topleft'):
        """
        image(np.ndarray): input image
        target_size(tuple): target size (width, height)
        align(str): alignment type, 'topleft' or 'center'
        Returns:
            np.ndarray: resized image
        """
        target_w, target_h = target_size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized_image = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        if align == 'topleft':
            x_offset, y_offset = 0, 0
        elif align == 'center':
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
        else:
            raise ValueError("align is 'topleft' or 'center'")

        padded_image[y_offset:y_offset + new_h,
                     x_offset:x_offset + new_w] = resized_image

        return padded_image

    def pack_data_without_pert(self, image_resize, batch_input_shape, ori_shape, text=None):
        assert image_resize.shape[:2] == batch_input_shape
        data_sample = DetDataSample()
        img_meta = {
            'img_shape':
            batch_input_shape,
            'ori_shape':
            ori_shape,
            'scale_factor': (batch_input_shape[0] / ori_shape[0],
                             batch_input_shape[1] / ori_shape[1])
        }
        data_sample.set_metainfo(img_meta)

        # 设置 Ground Truth
        gt_instances = InstanceData(metainfo=img_meta)
        gt_instances.bboxes = torch.rand((5, 4))
        gt_instances.labels = torch.tensor([1, 0, 1, 0, 1])
        data_sample.gt_instances = gt_instances

        if text is not None:
            # 设置文本信息
            data_sample.text = text

        data_tensor = torch.from_numpy(
            image_resize).permute((2, 0, 1)).cuda()
        data_dict = {'inputs': [data_tensor],
                     'data_samples': [data_sample]}
        return data_dict

    def pack_data_with_pert(self, image_resize, pert, batch_input_shape, ori_shape, text=None):
        assert image_resize.shape[:2] == batch_input_shape
        data_sample = DetDataSample()
        img_meta = {
            'img_shape':
            batch_input_shape,
            'ori_shape':
            ori_shape,
            'scale_factor': (batch_input_shape[0] / ori_shape[0],
                             batch_input_shape[1] / ori_shape[1])
        }
        data_sample.set_metainfo(img_meta)

        # 设置 Ground Truth
        gt_instances = InstanceData(metainfo=img_meta)
        gt_instances.bboxes = torch.rand((2, 4), device=self.device)
        gt_instances.labels = torch.tensor([1, 0], device=self.device)
        data_sample.gt_instances = gt_instances

        if text is not None:
            # 设置文本信息
            data_sample.text = text

        data_tensor = torch.from_numpy(image_resize).permute((2, 0, 1)).cuda()
        pert = F.interpolate(pert.unsqueeze(0), size=(
            data_tensor.size(1), data_tensor.size(2)), mode='bilinear').squeeze(0)

        data_tensor = (data_tensor + pert).clamp(0, 255)
        data_dict = {'inputs': [data_tensor], 'data_samples': [data_sample]}
        return data_dict

    def get_test_data(self, model, img, text=None):
        """get data format for testing
        Args:
            model(mmdet.models.detectors)): model
            img(np.ndarray): image
        """
        # init visualizer
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta

        width = img.shape[1]
        height = img.shape[0]
        ori_shape = (height, width)
        batch_input_shape = self.prefetch_batch_input_shape(
            model, (width, height))
        resize_wh = batch_input_shape[::-1]
        image_resize = self.resize_image_keep_ratio(
            img, resize_wh, align='topleft')
        data = self.pack_data_without_pert(
            image_resize, batch_input_shape, ori_shape, text)
        # print(data)
        # torch.save(data['inputs'][0], 'image2tensor.pt')
        with torch.no_grad():
            result = model.test_step(data)[0]
            # print(f"模型推理结果: {result.pred_instances['labels'].shape}")
            # result = model.test_step_(data)  # for test, faster_rcnn pass
            # print(f"模型推理结果: {result}")
            visualizer.add_datasample(
                name='results',
                image=img,
                data_sample=result,
                draw_gt=False,
                show=False,
                pred_score_thr=0.5,
                out_file='images/model_test_step.jpg')
            print(f"检测结果保存到: images/model_test_step.jpg ✅")
        return data

    def get_train_data(self, model, img, pert, text=None):
        """get data format for training
        Args:
            model(mmdet.models.detectors)): model
            img(np.ndarray): image
            pert(tensor): pertubation
        """
        # init visualizer
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta

        width = img.shape[1]
        height = img.shape[0]
        ori_shape = (height, width)
        batch_input_shape = self.prefetch_batch_input_shape(
            model, (width, height))
        resize_wh = batch_input_shape[::-1]
        image_resize = self.resize_image_keep_ratio(
            img, resize_wh, align='topleft')
        data = self.pack_data_with_pert(
            image_resize, pert, batch_input_shape, ori_shape, text)
        return data


def resize_bbox(bbox, original_size, new_size):
    """
    Resize the bounding box according to the new image size.

    Args:
    bbox (tuple): The original bounding box (x1, y1, x2, y2).
    original_size (tuple): The size (width, height) of the original image.
    new_size (tuple): The size (width, height) of the new image.

    Returns:
    tuple: The resized bounding box.
    """
    x1, y1, x2, y2 = bbox
    orig_width, orig_height = original_size
    new_width, new_height = new_size

    # Resize the bbox
    x1_new = x1 * new_width / orig_width
    y1_new = y1 * new_height / orig_height
    x2_new = x2 * new_width / orig_width
    y2_new = y2 * new_height / orig_height

    return torch.tensor([x1_new, y1_new, x2_new, y2_new])


def download_checkpoints():
    model_info = {
        'glip': {
            'config_file': 'mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py',
            'checkpoint_file': 'weights/glip/glip_tiny_a_mmdet-b3654169.pth',
            'download_link': 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth'
        },
        'groundingdino': {
            'config_file': 'mmdetection/configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py',
            'checkpoint_file': 'weights/groundingdino/groundingdino_swint_ogc_mmdet-822d7e9d.pth',
            'download_link': 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
        },
        'yolo-world': {
            'config_file': 'YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py',
            'checkpoint_file': 'weights/yolo_world/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth',
            'download_link': 'https://cdn-lfs-us-1.hf.co/repos/60/37/6037308e29abfdfd0058944e23eef6e0985d74bbb91fb2823de37a9ddfc4cc89/9babe3f6e2b73cd64f545b59de358fc7c47a85059cda9767f0a69f6bfae42d6d?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth%3B+filename%3D%22yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth%22%3B&Expires=1745299494&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0NTI5OTQ5NH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzYwLzM3LzYwMzczMDhlMjlhYmZkZmQwMDU4OTQ0ZTIzZWVmNmUwOTg1ZDc0YmJiOTFmYjI4MjNkZTM3YTlkZGZjNGNjODkvOWJhYmUzZjZlMmI3M2NkNjRmNTQ1YjU5ZGUzNThmYzdjNDdhODUwNTljZGE5NzY3ZjBhNjlmNmJmYWU0MmQ2ZD9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=J8HukWYPI5f5b0t-gm67oNKVMsBnw247iEKffTT9ImaZaUpCXtiORktV-ivF7221AFhTzBqueyzyHsTFmhz1TH6yBAJ2d9KgvaH8jYtoJEUs21RZPVpmw4U25qwzwNz4i-fAkBXirumZGc7TfX64HGES4krh1XZxJtmPdVLnVrJWCZzWg2psCWCeaZ8Se%7E%7ESzegXwGlSh4pSbSArEbcLa3aTlfWv1RudcvKoTM61xM3O9e1vtmO2YUyUujVtpAmmmc%7EoRb9jAQcuv1IZ6d3H2avy-RJBZrXQmSTe7838xncsf1LUKt-Dpaoy%7EpcbP-ji7Z-NcjwIzWTrqQi10qo42A__&Key-Pair-Id=K24J24Z295AEI9'
        }
    }

    checkpoints_root = Path('weights/download')
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    existing_files = list(checkpoints_root.glob('*.pth'))
    existing_files = [file.name for file in existing_files]

    for idx, model_name in enumerate(model_info):
        url = model_info[model_name]['download_link']
        file_name = url.split('/')[-1]
        if file_name in existing_files:
            print(f"{model_name} already exists, {idx+1}/{len(model_info)}")
            continue
        print(f'downloading {model_name} {idx+1}/{len(model_info)}')
        file_data = urllib.request.urlopen(url).read()
        with open(checkpoints_root / file_name, 'wb') as f:
            f.write(file_data)


def loss_smooth_img(img, mask):
    img = img / 255.0
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    mask = mask[:, :-1, :-1]
    mask = mask.unsqueeze(1)
    return 1e-4 * torch.sum(mask * (s1 + s2))


def loss_smooth(texture, device):
    texture = texture.permute(0, 4, 1, 2, 3)
    loss = 0.0
    loss += ((texture[:, :, 1:, :, :] - texture[:, :, :-1, :, :]) ** 2).mean()
    loss += ((texture[:, :, :, 1:, :] - texture[:, :, :, :-1, :]) ** 2).mean()
    loss += ((texture[:, :, :, :, 1:] - texture[:, :, :, :, :-1]) ** 2).mean()
    return torch.tensor(loss, device=device)


def loss_color(texture, device):
    target_colors = torch.tensor([
        [0.30, 0.45, 0.25],  # camouflage green
        [0.18, 0.28, 0.12],  # deep green
        [0.50, 0.50, 0.50],  # mid gray
        [0.65, 0.65, 0.65],  # light gray
        [0.40, 0.35, 0.28],  # mud brown
        [0.25, 0.20, 0.15],  # dark brown
        [0.15, 0.20, 0.10],  # dark olive green
        [0.35, 0.50, 0.30],  # soft army green
    ], device=device)
    tex_flat = texture.view(-1, 3)
    distances = torch.cdist(tex_flat, target_colors)
    min_dist = distances.min(dim=1)[0]
    return torch.tensor(min_dist.mean(), device=device)


def sort_iou(result, label, img_size=800, iou_thresh=0.3, score_thresh=0.1):
    pred_bboxes = result.pred_instances['bboxes']
    w, h = img_size, img_size

    def yolo_to_xyxy(box, img_w, img_h):
        x_c, y_c, w, h = box
        x1 = (x_c - w / 2) * img_w
        y1 = (y_c - h / 2) * img_h
        x2 = (x_c + w / 2) * img_w
        y2 = (y_c + h / 2) * img_h
        return [x1, y1, x2, y2]

    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
        return iou

    with open(label, 'r') as f:
        lines = f.readlines()

    gt_boxes = [yolo_to_xyxy(
        list(map(float, line.strip().split()[1:5])), w, h) for line in lines]

    pred_boxes = pred_bboxes.tolist()
    scores = result.pred_instances['scores'].tolist()
    best_pred_per_gt = {}  # key: gt_idx, value: (pred_idx, max_iou)
    for i, pred_box_xyxy in enumerate(pred_boxes):
        if scores[i] < score_thresh:
            continue

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(gt_box, pred_box_xyxy)
            if iou > iou_thresh:
                if gt_idx not in best_pred_per_gt or iou > best_pred_per_gt[gt_idx][1]:
                    best_pred_per_gt[gt_idx] = (i, iou)

    kept_indices = set()
    for pred_idx, _ in best_pred_per_gt.values():
        kept_indices.add(pred_idx)

    for i in range(len(result.pred_instances['scores'])):
        if i in kept_indices:
            result.pred_instances['scores'][i] = scores[i]
        else:
            result.pred_instances['scores'][i] = 0.0
    return result


def loss_target(result, label, img_size, pred_thresh, device, iou_thresh=0.3):
    pred_bboxes = result.pred_instances['bboxes'][result.pred_instances['scores'] > pred_thresh]
    pred_scores = result.pred_instances['scores'][result.pred_instances['scores'] > pred_thresh]
    w, h = img_size, img_size

    def yolo_to_xyxy(box, img_w, img_h):
        x_c, y_c, w, h = box
        x1 = (x_c - w / 2) * img_w
        y1 = (y_c - h / 2) * img_h
        x2 = (x_c + w / 2) * img_w
        y2 = (y_c + h / 2) * img_h
        return [x1, y1, x2, y2]

    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
        return iou

    with open(label, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        return torch.tensor(0.0, device=device)

    gt_boxes = [yolo_to_xyxy(
        list(map(float, line.strip().split()[1:5])), w, h) for line in lines]
    loss_scores = []
    for gt_box in gt_boxes:
        for pred_box, score in zip(pred_bboxes, pred_scores):
            iou = compute_iou(gt_box, pred_box.tolist())
            if iou > iou_thresh:
                loss_scores.append(score)
    if len(loss_scores) == 0:
        return torch.tensor(0.0, device=device)

    loss = torch.stack(loss_scores).mean()
    return loss


def calculate_ratio(mask):
    nonzero_counts = torch.sum(mask != 0, dim=(1, 2), dtype=torch.float)
    total_pixels = mask.size(1) * mask.size(2)
    ratios = nonzero_counts / total_pixels
    return ratios


def loss_mse(texture_origin, texture_param, device):
    criterion = torch.nn.MSELoss()
    loss = criterion(texture_origin, texture_param)
    return torch.tensor(loss, device=device)


def get_synonyms_wordnet(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)


if __name__ == "__main__":
    car_synonyms = get_synonyms_wordnet("car")
    print(car_synonyms)
