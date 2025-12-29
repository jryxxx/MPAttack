import pathlib
import os
import argparse
import sys

def batch_rename(directory_path: str):
    """
    批量重命名指定目录下的文件，将文件名中的 '.npz' 移除。
    例如：'pos_1_..._dist30_0.npz.png' 重命名为 'pos_1_..._dist30_0.png'
    """
    
    target_dir = pathlib.Path(directory_path)

    if not target_dir.is_dir():
        print(f"❌ 错误：路径 '{directory_path}' 不是一个有效的目录。")
        sys.exit(1)

    print(f"✅ 开始处理目录: {target_dir.resolve()}")
    
    # 使用 glob 查找所有匹配 '*.npz.png' 模式的文件
    files_to_rename = list(target_dir.glob('*.npz.png'))
    
    if not files_to_rename:
        print("⚠️ 目录中未找到任何以 '.npz.png' 结尾的文件。无需操作。")
        return

    success_count = 0
    
    for original_path in files_to_rename:
        # 1. 替换字符串中的 '.npz'
        # 例如：从 'pos_1_..._dist30_0.npz.png' 得到 'pos_1_..._dist30_0.png'
        new_name = original_path.name.replace('.npz', '')
        
        # 2. 构造新的完整路径
        new_path = original_path.with_name(new_name)
        
        # 3. 执行重命名操作
        try:
            # 使用 os.rename 或 pathlib.Path.rename()
            original_path.rename(new_path)
            print(f"   重命名成功: {original_path.name} -> {new_path.name}")
            success_count += 1
        except OSError as e:
            print(f"❌ 无法重命名文件 {original_path.name}: {e}")

    print("-" * 30)
    print(f"✨ 批量重命名完成。共处理 {len(files_to_rename)} 个文件，成功 {success_count} 个。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量重命名文件，将文件名中的 '.npz' 移除。例如 X.npz.png -> X.png"
    )
    parser.add_argument(
        "--directory", 
        default="/root/ovd-attack/test/images_val/textures_car_green",
        type=str, 
        help="包含要重命名文件的文件夹路径。"
    )
    
    args = parser.parse_args()
    batch_rename(args.directory)