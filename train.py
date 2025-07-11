#!/usr/bin/python3
# _*_ coding: utf-8 _*_

# @Date        : 2025/04/15 15:06
# @Author      : Ruiyang Jia
# @File        : train.py
# @Software    : Visual Studio Code
# @Description :

import warnings
import os
import argparse
import torch
from mmdet.apis import init_detector
import neural_renderer
import numpy as np
from utils.data_loader import DatasetAdv
from utils.utils import loss_target, loss_smooth, loss_smooth_img, loss_mse, loss_color
import tqdm
import wandb
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Adversarial Attacked Image Generation pipeline with MMDetection')
    parser.add_argument(
        '--config',
        # default='mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py' # 效果相差不大，可以更换其他模型
        # mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py
        # mmdetection/configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py # 效果可以
        # YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py
        default='mmdetection/configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py',
        help='model config file path')
    parser.add_argument(
        '--checkpoint',
        # default='weights/glip/glip_tiny_a_mmdet-b3654169.pth'
        # weights/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_180419-e6addd96.pth
        # weights/groundingdino/groundingdino_swint_ogc_mmdet-822d7e9d.pth
        # weights/yolo_world/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth
        default='weights/groundingdino/groundingdino_swint_ogc_mmdet-822d7e9d.pth',
        help='checkpoint file')
    parser.add_argument(
        '--data-path',
        default='/root/autodl-tmp/dataset/train',
        help='the directory to load the dataset')
    parser.add_argument(
        '--save-dir',
        default=f'images/adv_images/multi_prompt_train',
        help='the directory to save the generated images')
    parser.add_argument(
        '--device',
        default='cuda')
    parser.add_argument(
        '--thr',
        default=0.1,
        help='threshold to control model results')
    parser.add_argument(
        '--obj-file',
        default='data/test.obj',
        help='3d car model obj')
    parser.add_argument(
        '--epoches',
        default=5,
        help='epoches for optimized method')
    parser.add_argument(
        '--lr',
        default=0.01,
        help='learning rate for optimized method')
    parser.add_argument(
        '--faces',
        default='data/faces_new.txt',
        help='the face points of the 3d model')
    parser.add_argument(
        '--prompt-mode',
        choices=['single', 'multi'],
        help='prompt mode: single or multi')
    args = parser.parse_args()

    return args


def cal_texture__(texture_param, texture_origin, texture_mask, device, texture_content=None, content=False):
    # keep in green color
    min_color = torch.tensor([0.03, 0.08, 0.02], device=device)
    max_color = torch.tensor([0.25, 0.45, 0.18], device=device)

    if content:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    textures = textures * (max_color - min_color) + min_color
    textures = torch.clamp(textures, 0.0, 1.0)
    return texture_origin * (1 - texture_mask) + texture_mask * textures


def cal_texture_(texture_param, texture_origin, texture_mask, device, texture_content=None, content=False):
    # if use original texture, use this function
    if content:
        textures = texture_content.clamp(0, 1)
    else:
        textures = texture_param.clamp(0, 1)
    return texture_origin * (1 - texture_mask) + texture_mask * textures


def cal_texture(texture_param, texture_origin, texture_mask, device, texture_content=None, content=False):
    if content:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    return texture_origin * (1 - texture_mask) + texture_mask * textures


def collate_fn(batch):
    return batch


def main(args):

    # wandb
    # os.environ["WANDB_API_KEY"] = '7fc482dfa8c91872919d8e71d8d46fc6a361ef3f'
    # os.environ["WANDB_MODE"] = "offline"
    # wandb.init(project="OVD Attack", name="Train",
    #            settings=dict(init_timeout=120))

    CONFIG = args.config
    CHECKPOINT = args.checkpoint
    device = args.device
    output_path = args.save_dir
    conf_thresh = args.thr
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # if not os.path.exists(os.path.join(output_path, 'detect')):
    #     os.makedirs(os.path.join(output_path, 'detect'))
    # if not os.path.exists(os.path.join(output_path, 'image')):
    #     os.makedirs(os.path.join(output_path, 'image'))

    if 'glip' in CONFIG:
        model_type = 'glip'
    elif 'dino' in CONFIG:
        model_type = 'dino'
    else:
        model_type = 'yolo'

    # load model
    if model_type == 'yolo':
        import sys
        sys.path.append('/root/ovd-attack/YOLO-World')
        model = init_detector(CONFIG, CHECKPOINT, device=device)
    else:
        model = init_detector(CONFIG, CHECKPOINT, device=device)
    for param in model.parameters():
        param.requires_grad = False
    # load 3d model
    texture_size = 6
    vertices, faces, texture_origin = neural_renderer.load_obj(filename_obj=args.obj_file, texture_size=texture_size,
                                                               load_texture=True)
    # print(f'max: {texture_origin.max()}, min: {texture_origin.min()}')
    # np.save(f"textures/texture_origin.npy", texture_origin.data.cpu().numpy())

    # don't use mse loss
    # zero initialization
    # texture_param = np.zeros(
    #     (1, faces.shape[0], texture_size, texture_size, texture_size, 3)).astype('float32')
    # texture_param = torch.autograd.Variable(
    #     torch.from_numpy(texture_param).to(device), requires_grad=True)
    # optim = torch.optim.Adam([texture_param], lr=args.lr)
    # random initialization
    texture_param = np.random.random(
        (1, faces.shape[0], texture_size, texture_size, texture_size, 3)).astype('float32')
    texture_param = torch.autograd.Variable(
        torch.from_numpy(texture_param).to(device), requires_grad=True)
    optim = torch.optim.Adam([texture_param], lr=args.lr)

    # use mse loss
    # texture_param = torch.autograd.Variable(
    #     texture_origin.clone().detach().to(device), requires_grad=True)
    # optim = torch.optim.Adam([texture_param], lr=args.lr)

    # use perturbation
    # texture_pert = np.zeros(
    #     (1, faces.shape[0], texture_size, texture_size, texture_size, 3)).astype('float32')
    # texture_pert = torch.autograd.Variable(
    #     torch.from_numpy(texture_pert).to(device), requires_grad=True)
    # texture_param = texture_origin + texture_pert
    # optim = torch.optim.Adam([texture_pert], lr=args.lr)

    # load face points
    texture_mask = np.zeros(
        (faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
    with open(args.faces, 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            if face_id != '\n':
                texture_mask[int(face_id) - 1, :, :, :, :] = 1
    texture_mask = torch.from_numpy(texture_mask).to(device).unsqueeze(0)

    # set prompt
    if args.prompt_mode == 'single':
        texts_list = ["vehicle"]
    else:
        texts_list = ["vehicle", "car", "drive", "wheels"]

    # load dataset
    data_dir = os.path.join(args.data_path, 'npz/')
    img_size = 800
    mask_dir = os.path.join(args.data_path, 'mask/')
    label_dir = os.path.join(args.data_path, 'label/')
    ret_mask = True
    dataset = DatasetAdv(
        data_dir, args.prompt_mode, img_size, texture_size, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=ret_mask)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    sample_num = len(data_loader)
    textures = cal_texture(
        texture_param, texture_origin, texture_mask, device)
    dataset.set_textures(textures)
    loss_list = []

    # train
    print('#' * os.get_terminal_size().columns)
    print(f'Args: {args}')
    print('#' * os.get_terminal_size().columns)
    for epoch in range(1, args.epoches + 1):
        print('Epoch %d' % (epoch))
        pbar = enumerate(data_loader)
        pbar = tqdm.tqdm(pbar, total=sample_num)
        for i, sample in pbar:
            loss = 0
            data = sample[0]['data']
            mask = sample[0]['mask']
            label_name = sample[0]['filename'].split('.')[0] + '.txt'
            label_path = os.path.join(label_dir, label_name)

            mask_tensor = mask
            image_tensor = data['inputs'][0].unsqueeze(0)

            # save per 100 images
            if i % 100 == 0:
                print('Save images...')
                image_tensor_copy = data['inputs'][0].permute(1, 2, 0) / 255.0
                image_numpy = image_tensor_copy.detach().cpu().numpy()
                os.makedirs('test/images', exist_ok=True)
                plt.imsave(f"test/images/test_vis_{i}.png", image_numpy)

            # multi prompt
            for text in texts_list:
                if model_type == 'yolo':
                    import sys
                    from mmdet.structures import DetDataSample
                    sys.path.append('/root/ovd-attack/YOLO-World')
                    data_sample = DetDataSample()
                    img_meta = {
                        'img_shape': (800, 800, 3),
                        'ori_shape': (800, 800),
                        'scale_factor': (1.0, 1.0),
                        'texts': [text],
                    }
                    data_sample.set_metainfo(img_meta)
                    data_sample.text = text
                    data['data_samples'] = [data_sample]
                    data['inputs'] = data['inputs'][0].unsqueeze(0)
                    result = model.test_step(data)[0]

                    # use multi loss
                    # scores = result.pred_instances['scores'][result.pred_instances['scores'] > conf_thresh]
                    # a, b, c, d, e = 1, 0, 0, 0, 0
                    # if scores.numel() == 0:
                    #     loss_score_ = torch.tensor(0.0, device=device) * a
                    # else:
                    #     loss_score_ = scores.mean() * a
                    # loss_mse_ = loss_mse(
                    #     texture_origin, texture_param, device) * b
                    # loss_smooth_ = loss_smooth(texture_param, device) * c
                    # loss_color_ = loss_color(texture_param, device) * d
                    # loss_diff_ = torch.norm(
                    #     texture_param - texture_origin, p=2) * e
                    # loss += loss_score_ + loss_mse_ + loss_smooth_ + loss_color_ + loss_diff_

                    # use single loss
                    # scores = result.pred_instances['scores'][result.pred_instances['scores'] > conf_thresh]
                    # if scores.numel() == 0:
                    #     loss += torch.tensor(0.0, device=device)
                    # else:
                    #     loss += scores.mean()

                    # use iou loss
                    loss += loss_target(result, label_path,
                                        img_size, conf_thresh, device)

                else:
                    data['data_samples'][0].text = text
                    result = model.test_step(data)[0]

                    # use multi loss
                    # scores = result.pred_instances['scores'][result.pred_instances['scores'] > conf_thresh]
                    # a, b, c, d, e = 1, 0, 0, 0, 0
                    # if scores.numel() == 0:
                    #     loss_score_ = torch.tensor(0.0, device=device) * a
                    # else:
                    #     loss_score_ = scores.mean() * a
                    # loss_mse_ = loss_mse(
                    #     texture_origin, texture_param, device) * b
                    # loss_smooth_ = loss_smooth(texture_param, device) * c
                    # loss_color_ = loss_color(texture_param, device) * d
                    # loss_diff_ = torch.norm(
                    #     texture_param - texture_origin, p=2) * e
                    # loss += loss_score_ + loss_mse_ + loss_smooth_ + loss_color_ + loss_diff_

                    # use single loss
                    # scores = result.pred_instances['scores'][result.pred_instances['scores'] > conf_thresh]
                    # if scores.numel() == 0:
                    #     loss += torch.tensor(0.0, device=device)
                    # else:
                    #     loss += scores.mean()

                    # use iou loss
                    loss += loss_target(result, label_path,
                                        img_size, conf_thresh, device)

            if loss != 0:
                optim.zero_grad()
                loss.backward(retain_graph=False)
                optim.step()
            pbar.set_description('Loss %.8f' % (loss.data.cpu().numpy()))

            # use perturbation
            # pert = 0.2
            # texture_pert.data.clamp_(-pert, pert)
            # texture_param = texture_origin + texture_pert

            textures = cal_texture(
                texture_param, texture_origin, texture_mask, device)
            dataset.set_textures(textures)
            loss_list.append(loss.detach().cpu())

            # wandb
            # step = epoch * sample_num + i
            # wandb.log({"loss": loss.item(), "step": step})

        # save texture and loss every epoch
        os.makedirs(f'textures_iou/{model_type}', exist_ok=True)
        os.makedirs(f'results_iou/loss/{model_type}', exist_ok=True)
        np.save(f"textures_iou/{model_type}/texture_camouflage_{model_type}_{args.prompt_mode}_epoch_{epoch}.npy",
                texture_param.data.cpu().numpy())
        loss_tensor = torch.stack(loss_list)
        torch.save(
            loss_tensor, f'results_iou/loss/{model_type}/{model_type}_{args.prompt_mode}_losses.pt')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # VPN: source /etc/network_turbo
    # unVPN: unset http_proxy && unset https_proxy
    args = parse_args()

    def multi_train():
        model_list = ['dino', 'yolo', 'glip']
        prompt_list = ['single']
        for model in model_list:
            for prompt in prompt_list:
                if model == 'glip':
                    args.config = 'mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'
                    args.checkpoint = 'weights/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_180419-e6addd96.pth'
                elif model == 'dino':
                    args.config = 'mmdetection/configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py'
                    args.checkpoint = 'weights/groundingdino/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
                else:
                    args.config = 'YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'
                    args.checkpoint = 'weights/yolo_world/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth'
                args.prompt_mode = prompt
                main(args)
    multi_train()
