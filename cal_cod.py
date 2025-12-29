import os
import cv2
from tqdm import tqdm

import sod_metrics as M


def cal(image_root, method, image_name, f):
    FM = M.Fmeasure()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()

    if "car" in image_name:
        mask_root = '/root/ovd-attack/dataset_cod/masks_car'
    else:
        mask_root = '/root/ovd-attack/dataset_cod/masks_armored'

    image_size = (640, 640)
    mask_name_list = sorted(os.listdir(mask_root))

    for mask_name in tqdm(mask_name_list, desc=f"{method}-{image_name}", leave=False):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(image_root, mask_name)

        if not os.path.exists(pred_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.resize(mask, image_size)
        pred = cv2.resize(pred, image_size)

        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)

    fm = FM.get_results()['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']

    mean_em = "-" if em['curve'] is None else em['curve'].mean()

    f.write(
        f"{image_name:<25} | "
        f"Smeasure: {sm:.3f} | "
        f"meanEm: {mean_em if mean_em == '-' else round(mean_em, 3)} | "
        f"wFmeasure: {wfm:.3f} | "
        f"MAE: {mae:.3f}\n"
    )


if __name__ == "__main__":
    # images = [
    #     "textures_armored_brown",
    #     "textures_armored_green",
    #     "textures_car_brown",
    #     "textures_das_armored",
    #     "textures_das_car",
    #     "textures_fca_armored",
    #     "textures_fca_car",
    #     "armored",
    #     "car"
    # ]

    images = ["textures_car_green"]

    methods = ["feder", "mgl", "ugtr"]

    output_file = "/root/ovd-attack/dataset_cod/cod_results.txt"

    # 覆盖写入
    with open(output_file, "w") as f:
        for method in methods:
            f.write(f"\n========== Method: {method.upper()} ==========\n")
            for image in images:
                image_root = f"/root/ovd-attack/dataset_cod/results_{method}/{image}"
                cal(image_root, method, image, f)

    print(f"Results saved to {output_file}")
