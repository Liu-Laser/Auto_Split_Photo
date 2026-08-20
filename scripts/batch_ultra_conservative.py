#!/usr/bin/env python3
"""
batch_ultra_conservative.py - 批量超保守处理所有分割出的照片

使用方法：
python batch_ultra_conservative.py <输入目录>
"""
import os
import glob
from ultra_conservative_crop import ultra_conservative_crop

def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量超保守处理照片")
    parser.add_argument("input_dir", help="包含分割后照片的输入目录")
    parser.add_argument("-o", "--output_dir", help="输出目录（默认：input_dir + _ultra_conservative）")
    args = input_dir = parser.parse_args()

    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.input_dir + "_ultra_conservative"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有jpg文件
    input_dir = args.input_dir
    photo_files = glob.glob(os.path.join(input_dir, "*.jpg"))

    if not photo_files:
        print(f"在 {input_dir} 中没有找到jpg文件")
        return

    print(f"找到 {len(photo_files)} 张照片")

    # 处理每张照片
    for photo_path in photo_files:
        # 生成输出文件名
        basename = os.path.basename(photo_path)
        output_path = os.path.join(output_dir, basename)

        print(f"\n处理 {basename}...")
        ultra_conservative_crop(photo_path, output_path)
        print(f"完成: {output_path}")

    print(f"\n所有照片已处理完成，保存在: {output_dir}")

if __name__ == "__main__":
    main()