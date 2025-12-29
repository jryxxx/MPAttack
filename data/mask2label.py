import os
import cv2
import numpy as np


def mask_to_yolo_label(mask, class_id, image_width, image_height, mask_path):
    ys, xs = np.where(mask == 255)
    if len(xs) == 0 or len(ys) == 0:
        print(mask_path)
        return None

    x1, y1 = np.min(xs), np.min(ys)
    x2, y2 = np.max(xs), np.max(ys)

    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def process_all_masks(mask_dir, label_dir, class_id=0):
    os.makedirs(label_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(
        mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in mask_files:
        mask_path = os.path.join(mask_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to read {mask_path}")
            continue

        H, W = mask.shape
        label = mask_to_yolo_label(mask, class_id, W, H, mask_path)

        label_fname = os.path.splitext(fname)[0] + '.txt'
        label_path = os.path.join(label_dir, label_fname)

        if label is not None:
            with open(label_path, 'w') as f:
                f.write(label + '\n')
        else:
            open(label_path, 'w').close()


if __name__ == "__main__":
    mask_folder = '/media/bjh/disk-1.0TB/CARLA/carla/dataset/mask'
    label_folder = '/media/bjh/disk-1.0TB/CARLA/carla/dataset/label'

    process_all_masks(mask_folder, label_folder, class_id=0)
