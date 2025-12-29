import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.Network import Network
from utils.data_val import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def test(opt):
    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)

    model = Network(channels=96) 
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.pth_path).items()})
    model.cuda()
    model.eval()

    image_root = opt.test_dataset_path
    gt_root = image_root # 占位符，无实际用处
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        image = image.cuda()
        result = model(image)
        res = result[4]
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res*255)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=384, help='testing size')
    parser.add_argument('--pth_path', type=str, default='/root/ovd-attack/cod_methods/FEDER/checkpoints/checkpoint.pth') 
    parser.add_argument('--test_dataset_path', type=str, default='/root/ovd-attack/dataset_cod/imgs/') 
    parser.add_argument('--save_path', type=str, default='/root/ovd-attack/dataset_cod/results_feder/') 
    opt = parser.parse_args()

    # images = ["armored",
    #           "car",
    #           "textures_armored_brown", 
    #           "textures_armored_green", 
    #            "textures_car_brown",
    #            "textures_car_brown",
    #            "textures_das_armored",
    #            "textures_das_car",
    #            "textures_fca_armored",
    #            "textures_fca_car"]


    images = ["textures_car_green"]
    
    for image in images:
        opt.test_dataset_path = f'/root/ovd-attack/test/images_val/{image}/'
        opt.save_path = f'/root/ovd-attack/dataset_cod/results_feder/{image}/'
        test(opt)