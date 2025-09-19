# README

# DIRS
1. textures_iou_clamp/results_iou_clamp: MP-attack
2. textures_iou/results_iou: MP-attack / o-clamp
3. textures_clamp/results_clamp: MP-attack / o-iou
4. textures/results: SP-attack

## train
1. `python train.py`

## val
1. `python test.py` and get *.npy
2. `python cal_asr.py` and get ASR
3. `python get_car3d.py` and get images
4. `python cal_sal.py` and get saliency
