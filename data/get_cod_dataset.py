import os
import cv2
import argparse
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="准备 imgs / masks / edges 三个文件夹的数据")
    # parser.add_argument('--img_dir', type=str, default="/root/ovd-attack/test/images_val/textures_armored_brown")
    parser.add_argument('--img_dir', type=str, default="/root/ovd-attack/test/images_val/textures_car_green")
    # parser.add_argument('--mask_dir', type=str, default="/root/autodl-tmp/dataset/val/mask")
    parser.add_argument('--mask_dir', type=str, default="/root/autodl-tmp/data_tgrs2/test/mask")
    parser.add_argument('--out_root', type=str, default="/root/ovd-attack/dataset_cod")
    parser.add_argument('--canny_low', type=int, default=50)
    parser.add_argument('--canny_high', type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()

    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    out_root = Path(args.out_root)
    
    # out_masks = out_root / 'masks_armored'
    # out_edges = out_root / 'edges_armored'
    out_masks = out_root / 'masks_car'
    out_edges = out_root / 'edges_car'

    out_masks.mkdir(parents=True, exist_ok=True)
    out_edges.mkdir(parents=True, exist_ok=True)

    IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']

    # 将 kernel 改成 3x3（比 5x5 更细）
    kernel = np.ones((3, 3), np.uint8)

    img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    img_files = sorted(img_files)

    print(f"找到 {len(img_files)} 张图片。开始处理...")

    num_ok = 0
    num_skip = 0

    for img_path in img_files:
        name = img_path.stem

        candidate_masks = [
            mask_dir / f"{name}.png",
            mask_dir / f"{name}{img_path.suffix}",
        ]

        mask_path = None
        for cm in candidate_masks:
            if cm.is_file():
                mask_path = cm
                break

        if mask_path is None:
            print(f"[WARN] 找不到对应的 mask: {name}，跳过。")
            num_skip += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] 无法读取图片: {img_path}，跳过。")
            num_skip += 1
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] 无法读取mask: {mask_path}，跳过。")
            num_skip += 1
            continue

        if (img.shape[0] != mask.shape[0]) or (img.shape[1] != mask.shape[1]):
            print(f"[INFO] {name}: mask 尺寸与图片不一致，resize 一下。")
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # === 修改开始：改为形态学梯度(edge 1-2px 更细更连续) ===
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        # === 修改结束 ===

        out_mask_path = out_masks / f"{name}.png"
        out_edge_path = out_edges / f"{name}.png"

        cv2.imwrite(str(out_mask_path), mask)
        cv2.imwrite(str(out_edge_path), edge)

        num_ok += 1

    print(f"处理完成：成功 {num_ok} 个，跳过 {num_skip} 个。")
    print(f"masks 保存至: {out_masks}")
    print(f"edges 保存至: {out_edges}")


if __name__ == "__main__":
    main()
