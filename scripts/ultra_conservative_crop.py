#!/usr/bin/env python3
"""
ultra_conservative_crop.py - 专门用于超保守裁剪的工具

使用方法：
python ultra_conservative_crop.py <输入照片路径>
"""
import argparse
import sys
import numpy as np
from PIL import Image
import cv2

def ultra_conservative_crop(image_path, output_path):
    """超保守裁剪 - 仅裁剪纯白区域"""
    print(f"\n处理: {image_path}")

    # 加载图片
    pil_img = Image.open(image_path)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    arr = np.array(pil_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = arr.shape[:2]

    print(f"原始尺寸: {w}x{h}")

    # 使用极高的阈值（254）只检测纯白区域
    WHITE_THRESHOLD = 254
    row_white_ratio = np.mean(gray > WHITE_THRESHOLD, axis=1)
    col_white_ratio = np.mean(gray > WHITE_THRESHOLD, axis=0)

    # 四角分析
    corner_size = min(50, w // 10, h // 10)
    tl = np.mean(gray[:corner_size, :corner_size] > WHITE_THRESHOLD)
    tr = np.mean(gray[:corner_size, -corner_size:] > WHITE_THRESHOLD)
    bl = np.mean(gray[-corner_size:, :corner_size] > WHITE_THRESHOLD)
    br = np.mean(gray[-corner_size:, -corner_size:] > WHITE_THRESHOLD)

    diag_diff = max(abs(tl - br), abs(tr - bl))
    avg_corner_white = (tl + tr + bl + br) / 4

    print(f"四角白边(254): 左上={tl:.0%}, 右上={tr:.0%}, 左下={bl:.0%}, 右下={br:.0%}")
    print(f"对角线差异={diag_diff:.2f}, 平均={avg_corner_white:.0%}")

    # 仅在检测到大量纯白时才裁剪
    if avg_corner_white > 0.85:
        print("检测到大量纯白区域，进行最小裁剪")

        # 找到几乎完全是纯白的行和列
        pure_white_rows = np.where(row_white_ratio > 0.95)[0]
        pure_white_cols = np.where(col_white_ratio > 0.95)[0]

        if len(pure_white_rows) > 0 and len(pure_white_cols) > 0:
            # 计算裁剪边界，保留至少97%的内容
            y_start = pure_white_rows[0] if pure_white_rows[0] > 0 else 0
            y_end = pure_white_rows[-1] if pure_white_rows[-1] < h - 1 else h - 1
            x_start = pure_white_cols[0] if pure_white_cols[0] > 0 else 0
            x_end = pure_white_cols[-1] if pure_white_cols[-1] < w - 1 else w - 1

            # 确保不会裁剪太多
            height_ratio = (y_end - y_start + 1) / h
            width_ratio = (x_end - x_start + 1) / w

            if height_ratio > 0.97 and width_ratio > 0.97:
                cropped = arr[y_start:y_end + 1, x_start:x_end + 1]
                result = Image.fromarray(cropped)
                print(f"裁剪后尺寸: {result.width}x{result.height} (保留 {height_ratio*100:.1f}% x {width_ratio*100:.1f}% 内容)")
                result.save(output_path, quality=95, optimize=True)
                return result

    print("未检测到需要裁剪的纯白区域，保留完整照片")
    pil_img.save(output_path, quality=95, optimize=True)
    return pil_img

def main():
    parser = argparse.ArgumentParser(description="超保守照片裁剪工具")
    parser.add_argument("input", help="输入照片路径")
    parser.add_argument("-o", "--output", help="输出路径（默认：在输入文件后添加 _ultra_conservative）")
    args = parser.parse_args()

    input_path = args.input
    if not args.output:
        # 如果没有指定输出，添加后缀
        import os
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_ultra_conservative{ext}"
    else:
        output_path = args.output

    # 执行裁剪
    result = ultra_conservative_crop(input_path, output_path)
    print(f"\n保存到: {output_path}")

if __name__ == "__main__":
    main()