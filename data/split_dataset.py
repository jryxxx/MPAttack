import os
import random
import shutil

def split_dataset(src_dir, dst_dir, train_ratio=0.8):
    """
    将数据集按照比例划分为 train/test，包括：
    - npz/
    - mask/
    - label/
    所有文件按 base_name 一一对应。
    """

    # 输入路径
    npz_dir = os.path.join(src_dir, "npz")
    mask_dir = os.path.join(src_dir, "mask")
    label_dir = os.path.join(src_dir, "label")  # 注意：这里是 label 而不是 labels

    # 输出路径
    train_dir = os.path.join(dst_dir, "train")
    test_dir = os.path.join(dst_dir, "test")

    for d in [train_dir, test_dir]:
        os.makedirs(os.path.join(d, "npz"), exist_ok=True)
        os.makedirs(os.path.join(d, "mask"), exist_ok=True)
        os.makedirs(os.path.join(d, "label"), exist_ok=True)  # 这里也改为 label

    # 获取所有三者都存在的 base name
    valid_files = []
    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):
            continue
        base_name = os.path.splitext(f)[0]
        npz_path = os.path.join(npz_dir, base_name + ".npz")
        mask_path = os.path.join(mask_dir, base_name + ".png")
        label_path = os.path.join(label_dir, base_name + ".txt")
        if os.path.exists(npz_path) and os.path.exists(mask_path) and os.path.exists(label_path):
            valid_files.append(base_name)

    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * train_ratio)
    train_files = set(valid_files[:split_idx])
    test_files = set(valid_files[split_idx:])

    print(f"Total valid samples: {len(valid_files)}")
    print(f"Train: {len(train_files)}, Test: {len(test_files)}")

    # 复制函数
    def copy_pair(name, subset):
        src = {
            "npz": os.path.join(npz_dir, name + ".npz"),
            "mask": os.path.join(mask_dir, name + ".png"),
            "label": os.path.join(label_dir, name + ".txt")
        }
        dst = {
            "npz": os.path.join(dst_dir, subset, "npz", name + ".npz"),
            "mask": os.path.join(dst_dir, subset, "mask", name + ".png"),
            "label": os.path.join(dst_dir, subset, "label", name + ".txt")
        }

        for key in src:
            if os.path.exists(src[key]):
                shutil.copy(src[key], dst[key])
            else:
                print(f"[Skipped] File missing: {src[key]}")

    # 开始复制
    for name in train_files:
        copy_pair(name, "train")
    for name in test_files:
        copy_pair(name, "test")

    print("\n✅ Dataset split completed.")

# 示例调用
if __name__ == "__main__":
    dataset_dir = "/media/bjh/disk-1.0TB/CARLA/carla/dataset"     # 原始数据集路径
    output_dir = "/media/bjh/disk-1.0TB/CARLA/carla/data"       # 输出路径

    split_dataset(dataset_dir, output_dir, train_ratio=0.8)