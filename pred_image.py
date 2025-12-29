
from mmdet.structures import DetDataSample
from mmdet.apis import init_detector
import torch
from mmdet.registry import VISUALIZERS
import mmcv
import os
from utils.utils import sort_iou

def predict_image(model_type, image_path, save_path, texts):
    """
    inference the image
    """
    if model_type == 'glip':
        config = 'mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'
        checkpoint = '/root/autodl-tmp/weights/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_180419-e6addd96.pth'
    elif model_type == 'dino':
        config = 'mmdetection/configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py'
        checkpoint = '/root/autodl-tmp/weights/groundingdino/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
    else:
        config = 'YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'
        checkpoint = '/root/autodl-tmp/weights/yolo_world/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth'

    if  model_type == 'yolo':
        import sys
        sys.path.append('/root/ovd-attack/YOLO-World')
        model = init_detector(config, checkpoint)
        data_sample = DetDataSample()
        img_meta = {
            'img_shape': (800, 800, 3),
            'ori_shape': (800, 800),
            'scale_factor': (1.0, 1.0),
            'texts': [texts]
        }
        data_sample.set_metainfo(img_meta)
        data_sample.text = texts
        img = mmcv.imread(image_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        image_tensor = torch.from_numpy(
            img).permute((2, 0, 1)).cuda()
        data_dict = {'inputs': image_tensor.unsqueeze(0),
                        'data_samples': [data_sample]}
    else:
        model = init_detector(config, checkpoint)
        data_sample = DetDataSample()
        img_meta = {
            'img_shape': (800, 800),
            'ori_shape': (800, 800),
            'scale_factor': (1.0, 1.0),
        }
        data_sample.set_metainfo(img_meta)
        data_sample.text = texts
        img = mmcv.imread(image_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        image_tensor = torch.from_numpy(
            img).permute((2, 0, 1)).cuda()
        data_dict = {'inputs': [image_tensor],
                        'data_samples': [data_sample]}
    # save tensor
    # torch.save(data_dict['inputs']
    #            [0], 'detect_tensor/image2tensor.pt')
    result = model.test_step(data_dict)[0]
    label_name = os.path.splitext(image_path)[0] + '.txt'
    label_path = f"/root/autodl-tmp/dataset/val/label/{os.path.basename(label_name)}"
    iou_thresh, score_thresh = 0.3, 0.1
    result = sort_iou(result, label_path, 800, iou_thresh, score_thresh)
    with torch.no_grad():
        model.dataset_meta['classes'] = tuple([texts])
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta
        visualizer.add_datasample(
            name='results',
            image=data_dict['inputs'][0].permute(
                1, 2, 0).cpu().numpy(),
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=score_thresh,
            out_file=save_path)
        print(f"Save image to {save_path}")

if __name__ == '__main__':
    # VPN: source /etc/network_turbo
    # unVPN: unset http_proxy && unset https_proxy
    model_type = 'dino'  # glip, dino, yolo
    for texts in ["vehicle", "car", "drive", "wheels"]:
        image_path = 'test/images_val/textures_das_armored/pos_1_az180_0_el-60_0_dist30_0.png'
        save_path = f'test/images_res/{model_type}_{texts}.png'
        predict_image(model_type, image_path, save_path, texts)