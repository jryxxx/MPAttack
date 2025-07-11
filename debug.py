import numpy as np 
import cv2
import torch

if __name__ == "__main__":
    data = torch.load('results_sp/adv/glip/train_single/glip_multi_0.1.pt')
    print(data.keys())
    