import os
import random
import shutil

def move_half_images(src_folder, dst_folder, exts=('.jpg', '.png', '.jpeg', '.bmp', '.gif')):
    # 确保目标文件夹存在
    os.makedirs(dst_folder, exist_ok=True)

    # 获取源文件夹中所有图像文件
    files = [f for f in os.listdir(src_folder) if f.lower().endswith(exts)]

    # 打乱顺序并取一半
    random.shuffle(files)
    half_count = len(files) // 2
    selected_files = files[:half_count]

    # 移动文件
    for f in selected_files:
        src_path = os.path.join(src_folder, f)
        dst_path = os.path.join(dst_folder, f)
        shutil.move(src_path, dst_path)

    print(f"总共 {len(files)} 张图像，已随机移动 {len(selected_files)} 张到 {dst_folder}")

# 使用示例
src = "/media/zyserver/data16t/cailei/data/weather/allweather/raindrop"  # 源文件夹路径
dst = "/media/zyserver/data16t/cailei/data/weather/allweather/raindrop_semi"  # 目标文件夹路径
move_half_images(src, dst)