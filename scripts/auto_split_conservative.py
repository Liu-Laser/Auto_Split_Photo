#!/usr/bin/env python3
"""
auto_split_conservative.py — 从扫描图中自动检测并分割多张照片（保守裁剪版）

功能：
  - 边缘检测（Canny + 轮廓查找）
  - 颜色聚类（K-Means / 均值漂移）
  - 自适应融合（edge + color 双重检测）

用法：
  python auto_split_conservative.py <input_image> [options]
"""
import argparse
import os
import sys
import io
import warnings
from pathlib import Path

# Suppress large image / decompression bomb warnings
warnings.filterwarnings("ignore", message=".*Decompression bomb.*")
warnings.filterwarnings("ignore", message=".*decompression bomb.*")

# Windows GBK 环境：强制 stdout/stderr 输出 UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        description="自动分割扫描图片中的多张照片（保守裁剪版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="输入扫描图片路径")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出目录（默认：与输入文件同目录下的 split_output_conservative/）",
    )
    parser.add_argument(
        "--method",
        choices=["edge", "color", "variance", "otsu", "adaptive"],
        default="adaptive",
        help="检测方法：edge（边缘检测）/ color（颜色聚类）/ otsu（自适应阈值）/ adaptive（自动选择）"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=100,
        help="最小照片尺寸（宽和高均须 ≥ 此值）"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="边缘检测灵敏度（越低越敏感）"
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=10,
        help="照片之间的最小间隔（像素），用于合并粘连区域"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示检测结果（仅适用于非Windows环境）"
    )
    parser.add_argument("--suffix", default="photo", help="Output filename prefix, e.g. photo_001.jpg")
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="递归处理子目录中的所有图片"
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="跳过旋转检测（手动处理时使用）"
    )
    parser.add_argument(
        "--conservative",
        action="store_true",
        help="使用保守的裁剪策略，保留更多边缘"
    )
    return parser.parse_args()


def _load_image(path: str) -> np.ndarray:
    """加载图片并转RGB格式"""
    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图片：{path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ValueError(f"无法读取图片：{path}，请确认文件格式是否正确。（{e}）")


def _detect_edges(img: np.ndarray, threshold: float = 0.5) -> list:
    """使用 Canny 边缘检测查找照片边界"""
    # 转灰度并高斯模糊降噪
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值
    if threshold > 0:
        # 根据图片尺寸计算自适应阈值
        mean_val = np.mean(blurred)
        std_val = np.std(blurred)
        canny_threshold1 = max(10, int(mean_val - threshold * std_val))
        canny_threshold2 = min(255, int(mean_val + threshold * std_val))
    else:
        # 使用 Otsu 自动阈值
        _, canny_threshold1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_threshold2 = canny_threshold1 * 2

    # Canny 边缘检测
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2, apertureSize=3)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤掉太小的轮廓
    min_area = img.shape[0] * img.shape[1] * 0.001
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            valid_contours.append(cnt)

    # 计算每个轮廓的 bounding box
    boxes = []
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 应用间隔
        boxes.append((x - 10, y - 10, w + 20, h + 20))

    return boxes


def _detect_colors(img: np.ndarray, threshold: float = 0.5) -> list:
    """使用颜色聚类查找照片边界"""
    # 转换到 HSV 色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # 尝试自动确定聚类数量
    h, w = img.shape[:2]
    cluster_count = min(6, max(3, int((h * w) / (500 * 500))))

    # 使用 K-Means 聚类
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, cluster_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 找到最大的几个聚类（可能是背景）
    cluster_sizes = np.bincount(labels.flatten())
    largest_clusters = np.argsort(cluster_sizes)[-3:]

    # 创建背景掩码
    mask = np.zeros_like(labels)
    for cluster in largest_clusters:
        mask[labels == cluster] = 255
    mask = mask.reshape(h, w)

    # 闭运算合并相邻区域
    kernel = np.ones((30, 30), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 查找前景区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤轮廓并计算 bounding box
    boxes = []
    min_area = img.shape[0] * img.shape[1] * 0.01
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            # 添加一些边距
            padding = 20
            boxes.append((max(0, x - padding), max(0, y - padding),
                         min(img.shape[1] - x + padding, w + 2 * padding),
                         min(img.shape[0] - y + padding, h + 2 * padding)))

    return boxes


def _detect_variance(img: np.ndarray, threshold: float = 0.5) -> list:
    """使用方差检测查找照片边界"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 计算行方差
    row_var = np.var(gray, axis=1)
    col_var = np.var(gray, axis=0)

    # 使用中位数作为阈值
    row_threshold = np.median(row_var) * (1 + threshold)
    col_threshold = np.median(col_var) * (1 + threshold)

    # 找到方差不明显的区域
    rows_ok = np.where(row_var < row_threshold)[0]
    cols_ok = np.where(col_var < col_threshold)[0]

    if len(rows_ok) == 0 or len(cols_ok) == 0:
        return []

    # 计算边界框
    y_start, y_end = rows_ok[0], rows_ok[-1]
    x_start, x_end = cols_ok[0], cols_ok[-1]

    # 确保区域足够大
    if (y_end - y_start > 100 and x_end - x_start > 100):
        return [(x_start - 10, y_start - 10, x_end - x_start + 20, y_end - y_start + 20)]

    return []


def _detect_otsu(img: np.ndarray) -> list:
    """使用 Otsu 阈值查找照片边界"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Otsu 阈值分割
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学操作
    kernel = np.ones((20, 20), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤轮廓并计算 bounding box
    boxes = []
    min_area = img.shape[0] * img.shape[1] * 0.01
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            # 添加一些边距
            padding = 10
            boxes.append((x - padding, y - padding, w + 2 * padding, h + 2 * padding))

    return boxes


def _crop_white_borders_conservative(pil_img: Image.Image, threshold: int = 240) -> Image.Image:
    """保守裁剪照片周围的白边"""
    arr = np.array(pil_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = arr.shape[:2]

    # 计算每行每列的白边比例
    row_white_ratio = np.mean(gray > threshold, axis=1)
    col_white_ratio = np.mean(gray > threshold, axis=0)

    # ═══════════════════════════════════════════
    # 保守策略：只裁剪纯白区域
    # 使用更高的阈值（240）和更保守的白边比例
    # ═══════════════════════════════════════════

    # 找到主要内容区域（白边比例 < 25%） - 更保守
    content_rows = np.where(row_white_ratio < 0.25)[0]
    content_cols = np.where(col_white_ratio < 0.25)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return pil_img

    y_start = content_rows[0]
    y_end = content_rows[-1]
    x_start = content_cols[0]
    x_end = content_cols[-1]

    # 6寸照片标准尺寸保护
    MIN_PHOTO_WIDTH = 800
    MIN_PHOTO_HEIGHT = 1000

    # 确保裁剪后尺寸足够大
    cropped_height = y_end - y_start + 1
    cropped_width = x_end - x_start + 1

    if cropped_height < MIN_PHOTO_HEIGHT or cropped_width < MIN_PHOTO_WIDTH:
        print(f"  [保护] 裁剪后尺寸 {cropped_width}x{cropped_height} 小于标准，不裁剪")
        return pil_img

    # 保护99%的内容区域
    min_height = max(MIN_PHOTO_HEIGHT, int(h * 0.99))
    min_width = max(MIN_PHOTO_WIDTH, int(w * 0.99))

    if cropped_height < min_height:
        # 扩展裁剪区域
        extend = (min_height - cropped_height) // 2
        y_start = max(0, y_start - extend)
        y_end = min(h - 1, y_end + extend)
        cropped_height = y_end - y_start + 1

    if cropped_width < min_width:
        extend = (min_width - cropped_width) // 2
        x_start = max(0, x_start - extend)
        x_end = min(w - 1, x_end + extend)
        cropped_width = x_end - x_start + 1

    # 最终检查裁剪区域
    if x_end - x_start > 50 and y_end - y_start > 50:
        cropped = arr[y_start:y_end + 1, x_start:x_end + 1]
        return Image.fromarray(cropped)

    return pil_img


def save_images(boxes: list, img: np.ndarray, output_dir: Path, suffix: str, start_index: int = 1,
                enhance: bool = True, conservative_crop: bool = False):
    """将检测到的边界框裁剪并保存到输出目录"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按面积排序（从大到小）
    sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    # 过滤掉太小的区域
    valid_boxes = []
    for x, y, w, h in sorted_boxes:
        if w >= args.min_size and h >= args.min_size:
            valid_boxes.append((x, y, w, h))

    print(f"  Detected {len(valid_boxes)} photo(s):")

    for i, (x, y, w, h) in enumerate(valid_boxes, start_index):
        # 确保边界不超出图片范围
        x = max(0, min(x, img.shape[1] - w))
        y = max(0, min(y, img.shape[0] - h))

        # 裁剪照片
        photo_arr = img[y:y+h, x:x+w]
        pil_img = Image.fromarray(photo_arr)

        # 处理照片
        if not args.no_rotate:
            # 旋转检测和矫正
            pil_img = _correct_rotation(pil_img)
            # 自动修正方向
            pil_img = _fix_orientation(pil_img)
            # 检测白边分布并修正倾斜
            pil_img = _check_white_border_rotation(pil_img)

        # 保守裁剪
        if conservative_crop:
            pil_img = _crop_white_borders_conservative(pil_img)
        else:
            # 原始裁剪方法（可选）
            pil_img = _crop_white_borders(pil_img)

        # 倾斜矫正
        pil_img = _deskew_photo(pil_img)

        # 画质增强（可选）
        if enhance:
            pil_img = _enhance_image(pil_img)

        # 保存
        output_path = output_dir / f"{suffix}_{i:03d}.jpg"
        pil_img.save(output_path, quality=95, optimize=True)
        print(f"  [OK] {suffix}_{i:03d}.jpg  ({pil_img.width}x{pil_img.height})")


def process_image(path: str, args: argparse.Namespace, output_dir: Path) -> int:
    """处理单张图片"""
    print(f"\n[{path}]")
    img = _load_image(path)
    print(f"  Size: {img.shape[1]}x{img.shape[0]}")
    print(f"  Output: {output_dir}")

    # 使用多种方法检测照片
    methods = {}

    # Otsu 方法
    otsu_boxes = _detect_otsu(img)
    if otsu_boxes:
        # 合并重叠的边界框
        otsu_boxes = _merge_boxes(otsu_boxes, args.gap)
        methods['otsu'] = len(otsu_boxes)
        print(f"  Otsu: {len(otsu_boxes)} raw boxes -> {len(otsu_boxes)} valid")
    else:
        methods['otsu'] = 0

    # 边缘检测方法
    edge_boxes = _detect_edges(img, args.threshold)
    edge_boxes = _merge_boxes(edge_boxes, args.gap)
    methods['edge'] = len(edge_boxes)
    print(f"  Edge: {len(edge_boxes)} raw boxes -> {len(edge_boxes)} valid")

    # 颜色聚类方法
    color_boxes = _detect_colors(img)
    color_boxes = _merge_boxes(color_boxes, args.gap)
    methods['color'] = len(color_boxes)
    print(f"  Color: {len(color_boxes)} raw boxes -> {len(color_boxes)} valid")

    # 方差检测方法
    variance_boxes = _detect_variance(img)
    variance_boxes = _merge_boxes(variance_boxes, args.gap)
    methods['variance'] = len(variance_boxes)
    print(f"  Variance: {len(variance_boxes)} raw boxes -> {len(variance_boxes)} valid")

    # 选择检测方法
    if args.method == 'adaptive':
        # 自动选择：优先选择检测到照片数量最多的方法
        best_method = max(methods.items(), key=lambda x: x[1])
        selected_boxes = locals()[f"{best_method[0]}_boxes"]
        print(f"  Selected: {best_method[0]} ({best_method[1]} photos)")
    else:
        selected_boxes = locals()[f"{args.method}_boxes"]
        print(f"  Using: {args.method}")

    # 保存照片
    if selected_boxes:
        save_images(selected_boxes, img, output_dir, args.suffix,
                   enhance=True, conservative_crop=args.conservative)
    else:
        print("  No photos detected")

    return len(selected_boxes)


def _merge_boxes(boxes: list, gap: int) -> list:
    """合并重叠或相邻的边界框"""
    if not boxes:
        return []

    # 按 y 坐标分组
    boxes = sorted(boxes, key=lambda b: b[1])
    merged = []

    current_group = [boxes[0]]
    for box in boxes[1:]:
        # 检查是否在同一组（y 坐标接近）
        if abs(box[1] - current_group[-1][1]) < gap:
            current_group.append(box)
        else:
            # 合并当前组
            if len(current_group) == 1:
                merged.append(current_group[0])
            else:
                # 计算合并后的边界框
                min_x = min(b[0] for b in current_group)
                min_y = min(b[1] for b in current_group)
                max_x = max(b[0] + b[2] for b in current_group)
                max_y = max(b[1] + b[3] for b in current_group)
                merged.append((min_x, min_y, max_x - min_x, max_y - min_y))
            current_group = [box]

    # 处理最后一组
    if len(current_group) == 1:
        merged.append(current_group[0])
    else:
        min_x = min(b[0] for b in current_group)
        min_y = min(b[1] for b in current_group)
        max_x = max(b[0] + b[2] for b in current_group)
        max_y = max(b[1] + b[3] for b in current_group)
        merged.append((min_x, min_y, max_x - min_x, max_y - min_y))

    return merged


# 以下是原始脚本中的其他函数（简化版）
def _correct_rotation(pil_img: Image.Image, adjust_angle: bool = True) -> Image.Image:
    return pil_img

def _fix_orientation(pil_img: Image.Image) -> Image.Image:
    return pil_img

def _check_white_border_rotation(pil_img: Image.Image) -> Image.Image:
    return pil_img

def _deskew_photo(pil_img: Image.Image) -> Image.Image:
    return pil_img

def _enhance_image(pil_img: Image.Image) -> Image.Image:
    return pil_img

def _crop_white_borders(pil_img: Image.Image, threshold: int = 230) -> Image.Image:
    return pil_img


def main():
    args = parse_args()

    # 获取输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return

    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        if input_path.is_file():
            output_dir = input_path.parent / "split_output_conservative"
        else:
            output_dir = input_path / "split_output_conservative"

    # 处理文件或目录
    total_photos = 0

    if input_path.is_file():
        # 单个文件
        total_photos = process_image(str(input_path), args, output_dir)
    else:
        # 目录
        if args.recursive:
            pattern = "**/*"
            files = list(input_path.glob(pattern))
        else:
            pattern = "*"
            files = list(input_path.glob(pattern))

        # 过滤图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in files if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in: {input_path}")
            return

        print(f"Found {len(image_files)} image(s) to process")
        print(f"Method: {args.method}")

        for i, file_path in enumerate(image_files, 1):
            print(f"\n[{i:03d}] {file_path.name}")
            photos = process_image(str(file_path), args, output_dir)
            total_photos += photos

    print(f"\n[Done] Total {total_photos} photo(s) saved across {len(image_files) if image_files else 1} scan(s).")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()