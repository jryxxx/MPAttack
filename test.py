#!/usr/bin/python3
# _*_ coding: utf-8 _*_

# @Date        : 2025/04/15 15:06
# @Author      : Ruiyang Jia
# @File        : test.py
# @Software    : Visual Studio Code
# @Description :

from mmdet.structures import DetDataSample
from mmdet.apis import init_detector
import torch
from mmdet.registry import VISUALIZERS
import mmcv
from utils.utils import GetTrainData, sort_iou
from utils.data_loader import DatasetAdv, ApplyTexture, convert_npz_to_png
from train import collate_fn
import os
import numpy as np
import tqdm
import json
from glob import glob
import cv2
import shutil
import random
import neural_renderer
from torch.utils.data import DataLoader
import argparse


class TestAndVal():
    def __init__(self, texts, threshold, config, checkpoint):
        # If the same words exist in the previous and next categories
        # it will cause a category matching error and output the unobject category.
        self.texts = texts
        self.threshold = threshold
        self.config = config
        self.checkpoint = checkpoint
        if 'glip' in self.config:
            self.model_type = 'glip'
        elif 'dino' in self.config:
            self.model_type = 'dino'
        else:
            self.model_type = 'yolo'

    def predict_image(self, image_path, save_path, prompt='single'):
        """
        inference the image
        """
        self.texts = prompt
        if self.texts == 'single':
            texts = "vehicle"
        else:
            texts = "car", "vehicle", "drive", "wheels"
        if self.model_type == 'yolo':
            import sys
            sys.path.append('/root/ovd-attack/YOLO-World')
            model = init_detector(self.config, self.checkpoint)
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
            model = init_detector(self.config, self.checkpoint)
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

    def predict_folder(self, image_dir, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        image_paths = glob(os.path.join(image_dir, '*.png'))
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            output_path = os.path.join(save_dir, filename)
            try:
                self.predict_image(image_path, output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Failed: {filename}, error: {e}")

    def predict_folder_with_dets(self, image_dir, save_dir):
        """
        # inference the image folder
        usage:
        test.predict_folder('/root/autodl-tmp/dataset/test/images',
                            'images/clean_images_det')
        """
        model = init_detector(self.config, self.checkpoint)
        os.makedirs(save_dir, exist_ok=True)

        for filename in tqdm.tqdm(os.listdir(image_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            image_path = os.path.join(image_dir, filename)
            save_path = os.path.join(save_dir, filename)

            img = mmcv.imread(image_path)
            img = mmcv.imconvert(img, 'bgr', 'rgb')

            data_sample = DetDataSample()
            img_meta = {
                'img_shape': (800, 800),
                'ori_shape': (800, 800),
                'scale_factor': (1.0, 1.0)
            }
            data_sample.set_metainfo(img_meta)
            data_sample.text = self.texts

            image_tensor = torch.from_numpy(img).permute((2, 0, 1)).cuda()
            data_dict = {'inputs': [image_tensor],
                         'data_samples': [data_sample]}

            result = model.test_step(data_dict)[0]
            result_socres = result.pred_instances['scores'][result.pred_instances['scores'] > self.threshold]
            if result_socres.shape[0] > 0:
                is_save = True
            else:
                is_save = False
            with torch.no_grad():
                if is_save:
                    visualizer = VISUALIZERS.build(model.cfg.visualizer)
                    visualizer.dataset_meta = model.dataset_meta
                    visualizer.add_datasample(
                        name=filename,
                        image=image_tensor.permute(1, 2, 0).cpu().numpy(),
                        data_sample=result,
                        draw_gt=False,
                        show=False,
                        pred_score_thr=self.threshold,
                        out_file=save_path
                    )

    def apply_camouflage_texture(self, image_name, textures_file):
        """
        apply camouflage texture to the image
        """
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
        dataset = ApplyTexture(
            data_dir, self.texts, image_size, texture_size, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=True)

        def cal_texture(texture_content):
            textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
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

        textures_adv = cal_texture(texture_content_adv)
        dataset.set_textures(textures_adv)
        data_dict = dataset.apply(image_name)['img']
        # save adv image
        img_np = data_dict['inputs'][0].permute(
            1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imwrite(
            '/root/ovd-attack/test/pos_0_az270_0_el-90_0_dist30_0.png', img_np[:, :, ::-1])
        return data_dict

    def val_asr(self, textures_file):
        """
        usage
        """
        def yolo_to_xyxy(box, img_w, img_h):
            x_c, y_c, w, h = box
            x1 = int((x_c - w / 2) * img_w)
            y1 = int((y_c - h / 2) * img_h)
            x2 = int((x_c + w / 2) * img_w)
            y2 = int((y_c + h / 2) * img_h)
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
            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
            return iou

        iou_threshold = 0.3  # select more ious
        model_name = self.auto_model_name(self.config)
        model_train = self.auto_model_train(textures_file)
        if model_name == 'yolo':
            import sys
            sys.path.append('/root/ovd-attack/YOLO-World')
            model = init_detector(self.config, self.checkpoint)
        else:
            model = init_detector(self.config, self.checkpoint)
        # load 3d model
        texture_size = 6
        image_size = 800
        obj_file = 'data/test.obj'
        vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, texture_size=texture_size,
                                                             load_texture=True)
        # Camouflage Textures
        faces_file = 'data/faces_new.txt'
        texture_content_adv = torch.from_numpy(
            np.load(textures_file)).cuda(device=0)
        texture_origin = textures[None, :, :, :, :, :].cuda(device=0)
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
        label_dir = '/root/autodl-tmp/dataset/val/label/'
        dataset = DatasetAdv(
            data_dir, self.texts, image_size, texture_size, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=True)
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )

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

        textures_adv = cal_texture(texture_content_adv)
        dataset.set_textures(textures_adv)
        if self.texts == 'single':
            self.texts_list = ["vehicle"]
        else:
            # 4 prompts
            self.texts_list = ["vehicle", "car", "drive", "wheels"]
            # 8 prompts
            # self.texts_list = ["vehicle", "car", "drive", "wheels", "transport", "auto", "automobile", "motor"]
            # 12 prompts
            # self.texts_list = ["vehicle", "car", "drive", "wheels", "transport", "auto", "automobile", "motor",
            #                   "truck", "van", "bus", "sedan"]
        results_dict = {
            text: {
                'clean': {},
                'adv': {}
            }
            for text in self.texts_list
        }

        os.makedirs(f'results_iou/adv/{model_name}/{model_train}', exist_ok=True)
        pt_save_path = f'results_iou/adv/{model_name}/{model_train}/{model_name}_{self.texts}_{self.threshold}.pt'
        torch.save(results_dict, pt_save_path)
        print(f'Results have been saved to {pt_save_path}!')

        for sample in tqdm.tqdm(loader):
            for text in self.texts_list:
                filename = sample[0]['filename']
                label_path = os.path.join(
                    label_dir, os.path.splitext(filename)[0] + '.txt')
                # clean image
                img = sample[0]['img_clean']
                img = np.transpose(img, (2, 0, 1))
                img = np.resize(
                    img, (1, img.shape[0], img.shape[1], img.shape[2]))
                img = torch.from_numpy(img).cuda(device=0)
                if self.model_type == 'yolo':
                    import sys
                    sys.path.append('/root/ovd-attack/YOLO-World')
                    data_sample = DetDataSample()
                    img_meta = {
                        'img_shape': (800, 800),
                        'ori_shape': (800, 800),
                        'scale_factor': (1.0, 1.0),
                        'texts': [text]
                    }
                    data_sample.set_metainfo(img_meta)
                    data_sample.text = text
                    data_dict = {'inputs': img,
                                 'data_samples': [data_sample]}
                    result_clean = model.test_step(data_dict)[0]
                else:
                    data_sample = DetDataSample()
                    img_meta = {
                        'img_shape': (800, 800),
                        'ori_shape': (800, 800),
                        'scale_factor': (1.0, 1.0)
                    }
                    data_sample.set_metainfo(img_meta)
                    data_sample.text = text
                    data_dict = {'inputs': [img.squeeze(0)],
                                 'data_samples': [data_sample]}
                    result_clean = model.test_step(data_dict)[0]
                result_clean_det = result_clean.pred_instances['bboxes'][
                    result_clean.pred_instances['scores'] > self.threshold].detach().cpu().numpy()
                # adv image
                if self.model_type == 'yolo':
                    import sys
                    sys.path.append('/root/ovd-attack/YOLO-World')
                    data_dict = sample[0]['data']
                    data_sample = DetDataSample()
                    img_meta = {
                        'img_shape': (800, 800),
                        'ori_shape': (800, 800),
                        'scale_factor': (1.0, 1.0),
                        'texts': [text]
                    }
                    data_sample.set_metainfo(img_meta)
                    data_sample.text = text
                    data_dict['data_samples'] = [data_sample]
                    data_dict['inputs'] = data_dict['inputs'][0].unsqueeze(0)
                    result_adv = model.test_step(data_dict)[0]
                else:
                    data_dict = sample[0]['data']
                    data_dict['data_samples'][0].text = text
                    result_adv = model.test_step(data_dict)[0]
                result_adv_det = result_adv.pred_instances['bboxes'][
                    result_adv.pred_instances['scores'] > self.threshold].detach().cpu().numpy()

                # cal iou
                h, w = 800, 800
                results_dict[text]['clean'][filename] = False
                results_dict[text]['adv'][filename] = False
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    parts = list(map(float, line.strip().split()[1:5]))
                    gt_box = yolo_to_xyxy(parts, w, h)
                    for pred_box in result_clean_det:
                        iou = compute_iou(gt_box, pred_box)
                        if iou > iou_threshold:
                            results_dict[text]['clean'][filename] = True
                    for pred_box in result_adv_det:
                        iou = compute_iou(gt_box, pred_box)
                        if iou > iou_threshold:
                            results_dict[text]['adv'][filename] = True

        os.makedirs(f'results_iou/adv/{model_name}/{model_train}', exist_ok=True)
        pt_save_path = f'results_iou/adv/{model_name}/{model_train}/{model_name}_{self.texts}_{self.threshold}.pt'
        torch.save(results_dict, pt_save_path)
        print(f'Results have been saved to {pt_save_path}!')

    def auto_model_name(self, config):
        if "glip" in config:
            model = "glip"
        elif "dino" in config:
            model = "dino"
        else:
            model = "yolo"
        return model

    def auto_model_train(self, textures):
        if "single" in textures:
            mode = "train_single"
        else:
            mode = "train_multi"
        return mode

    @staticmethod
    def cal_asr_all(pt_file_path):
        """
        {
            "car": {"clean": {...}, "adv": {...}},
            "automobile": {"clean": {...}, "adv": {...}},
            ...
        }
        """
        results_dict = torch.load(pt_file_path)

        clean_result = {}
        adv_result = {}

        for text, result in results_dict.items():
            clean_dict = result.get("clean", {})
            adv_dict = result.get("adv", {})

            for filename, v in clean_dict.items():
                clean_result[filename] = clean_result.get(filename, False) or v

            for filename, v in adv_dict.items():
                adv_result[filename] = adv_result.get(filename, False) or v

        clean_true_count = sum(1 for v in clean_result.values() if v)
        adv_true_count = sum(1 for v in adv_result.values() if v)

        asr = (clean_true_count - adv_true_count) / max(clean_true_count, 1)

        print(f"Clean detections: {clean_true_count}")
        print(f"Adv detections: {adv_true_count}")
        print(f"ASR: {asr:.2%}")

        return clean_true_count, adv_true_count, asr

    @staticmethod
    def cal_asr_for_category(pt_file_path, category_name):
        """
        Args:
            pt_file_path (str)
            category_name (str)
        return:
            clean_true_count (int)
            adv_true_count (int)
            asr (float)
        """
        results_dict = torch.load(pt_file_path)

        if category_name not in results_dict:
            raise ValueError(
                f"Class {category_name} can't be found, available classes: {list(results_dict.keys())}")

        clean_dict = results_dict[category_name].get("clean", {})
        adv_dict = results_dict[category_name].get("adv", {})

        clean_true_count = sum(1 for v in clean_dict.values() if v)
        adv_true_count = sum(1 for v in adv_dict.values() if v)

        asr = (clean_true_count - adv_true_count) / max(clean_true_count, 1)

        print(f"[{category_name}] Clean detections: {clean_true_count}")
        print(f"[{category_name}] Adv detections: {adv_true_count}")
        print(f"[{category_name}] ASR: {asr:.2%}")

        return clean_true_count, adv_true_count, asr


def test_yolo_world(image_path, model_type):
    import sys
    sys.path.append('/root/ovd-attack/YOLO-World')
    if model_type == 'l':
        config = 'YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'
        checkpoint = 'weights/yolo_world/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth'
    else:
        config = 'YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py'
        checkpoint = 'weights/yolo_world_x/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth'

    model = init_detector(config, checkpoint)
    texts = ['car', 'transport', 'vehicle', 'auto', 'drive']
    for text in texts:
        text = [text]
        data_sample = DetDataSample()
        img_meta = {
            'img_shape': (800, 800, 3),
            'ori_shape': (800, 800),
            'scale_factor': (1.0, 1.0),
            'texts': text
        }
        data_sample.set_metainfo(img_meta)
        data_sample.text = text
        img = mmcv.imread(image_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        image_tensor = torch.from_numpy(
            img).permute((2, 0, 1)).cuda()
        data_dict = {'inputs': image_tensor.unsqueeze(0),
                     'data_samples': [data_sample]}
        with torch.no_grad():
            model.dataset_meta['classes'] = tuple(text)
            print(model.dataset_meta['classes'])
            result = model.test_step(data_dict)[0]
            visualizer = VISUALIZERS.build(model.cfg.visualizer)
            visualizer.dataset_meta = model.dataset_meta
            visualizer.add_datasample(
                name='results',
                image=data_dict['inputs'][0].permute(
                    1, 2, 0).cpu().numpy(),
                data_sample=result,
                draw_gt=False,
                show=False,
                pred_score_thr=0.5,
                out_file=f"test/yolo_world_res_{text}.png")
            print(f"Save image to test/yolo_world_res_{text}.png")


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        '--texts-mode', type=str, default='multi', help='prompt texts')
    argparse.add_argument(
        '--theroshold', type=float, default=0.1, help='threshold')
    argparse.add_argument(
        # mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py
        # mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py
        # mmdetection/configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py
        # YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py
        '--config', type=str, default='mmdetection/configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py', help='config file')
    argparse.add_argument(
        # weights/glip/glip_tiny_a_mmdet-b3654169.pth
        # weights/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_180419-e6addd96.pth
        # weights/groundingdino/groundingdino_swint_ogc_mmdet-822d7e9d.pth
        # weights/yolo_world/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth
        '--checkpoint', type=str, default='weights/groundingdino/groundingdino_swint_ogc_mmdet-822d7e9d.pth', help='checkpoint file')
    argparse.add_argument(
        '--clean-image-dir', type=str, default='/root/autodl-tmp/data/phy_attack/test', help='clean image dir')
    argparse.add_argument(
        '--adv-image-dir', type=str, default='images/adv_images/multi_prompt_train/image', help='adv image dir')
    argparse.add_argument(
        # textures/glip/texture_camouflage_forest_multi_epoch_5.npy
        # textures/dino/texture_camouflage_dino_forest_single_epoch_5.npy
        '--textures-file', type=str, default='textures/glip/texture_camouflage_glip_forest_single_epoch_5.npy', help='textures file')
    args = argparse.parse_args()

    # cal asr for all models
    def cal_asr_all_models():
        models = ['glip', 'dino', 'yolo']
        textures = ['single']
        for model in models:
            if model == 'glip':
                therosholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            elif model == 'dino':
                therosholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            else:
                therosholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            for theroshold in therosholds:
                for texture in textures:
                    print(
                        f"model: {model}, texture: {texture}, theroshold: {theroshold}")
                    args.theroshold = theroshold
                    if model == 'glip':
                        args.config = 'mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'
                        args.checkpoint = 'weights/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_180419-e6addd96.pth'
                        if texture == 'single':
                            args.textures_file = 'textures_iou_clamp/glip/texture_camouflage_glip_single_epoch_5.npy'
                        else:
                            args.textures_file = 'textures_iou_clamp/glip/texture_camouflage_glip_multi_epoch_5.npy'
                    elif model == 'dino':
                        args.config = 'mmdetection/configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py'
                        args.checkpoint = 'weights/groundingdino/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
                        if texture == 'single':
                            args.textures_file = 'textures_iou_clamp/dino/texture_camouflage_dino_single_epoch_5.npy'
                        else:
                            args.textures_file = 'textures_iou_clamp/dino/texture_camouflage_dino_multi_epoch_5.npy'
                    else:
                        args.config = 'YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'
                        args.checkpoint = 'weights/yolo_world/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth'
                        if texture == 'single':
                            args.textures_file = 'textures_iou_clamp/yolo/texture_camouflage_yolo_single_epoch_5.npy'
                        else:
                            args.textures_file = 'textures_iou_clamp/yolo/texture_camouflage_yolo_multi_epoch_5.npy'
                    test = TestAndVal(args.texts_mode, args.theroshold,
                                      args.config, args.checkpoint)
                    test.val_asr(args.textures_file)

    cal_asr_all_models()

    # init
    # test = TestAndVal(
    #     args.texts_mode, args.theroshold, args.config, args.checkpoint)

    # save npz to image
    # save_npz_to_image(
    #     '/root/autodl-tmp/dataset/train/npz/data0.npz', 'test/clean_image.png')

    # test yolo world
    # test_yolo_world('test/clean_image.png', 'x')

    # cal asr
    # test.val_asr(args.textures_file)

    # apply texture
    # test.apply_camouflage_texture(
    #     image_name='pos_0_az270_0_el-90_0_dist30_0.npz', textures_file='textures_iou/dino/texture_camouflage_dino_multi_epoch_5.npy')

    # convert npz to png
    # convert_npz_to_png(
    #     '/root/autodl-tmp/dataset/val/npz', 'test/images')

    # test image
    # test.predict_folder(
    #     '/root/autodl-tmp/dataset/val/npz', 'test/test_clean')
    # test.predict_image(
    #     'test/pos_0_az270_0_el-90_0_dist30_0.png', 'test/test_vis_res.png')
    # test.predict_folder('test/test/images/images', 'test/test/images/res')
    # test.predict_image_specific_texts(image_name='data10927.npz', textures_file='textures/dino/texture_camouflage_dino_forest_multi_epoch_4.npy', texts='transport', mode='forest', save_path='test/res.png')
