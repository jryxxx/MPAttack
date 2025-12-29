import cv2
import numpy as np
import os
import glob

def visualize_mask_and_label(mask_path, label_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = map(float, parts)

            x_center *= W
            y_center *= H
            width *= W
            height *= H

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            cv2.rectangle(mask_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(mask_color, f"Class {int(class_id)}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        print(f"Label file not found: {label_path}")

    return mask_color


def interactive_view(mask_dir, label_dir):
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    idx = 0

    while True:
        if idx < 0:
            idx = 0
        if idx >= len(mask_files):
            idx = len(mask_files) - 1

        mask_path = mask_files[idx]
        filename = os.path.basename(mask_path)
        label_path = os.path.join(label_dir, filename.replace(".png", ".txt"))

        img = visualize_mask_and_label(mask_path, label_path)

        cv2.imshow("Mask + Label Viewer (← → to navigate, q to quit)", img)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break
        elif key == ord('a'):   # a = 上一张
            idx -= 1
        elif key == ord('b'):   # b = 下一张
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    mask_dir = "/Users/ryjia/Downloads/data_tgrs2/train/mask"
    label_dir = "/Users/ryjia/Downloads/data_tgrs2/train/label"
    interactive_view(mask_dir, label_dir)
