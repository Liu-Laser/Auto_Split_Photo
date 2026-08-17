#!/usr/bin/env python3
"""
auto_split.py — 从扫描图中自动检测并分割多张照片

功能：
  - 边缘检测（Canny + 轮廓查找）
  - 颜色聚类（K-Means / 均值漂移）
  - 自适应融合（edge + color 双重检测）

用法：
  python auto_split.py <input_image> [options]

示例：
  python auto_split.py scan.jpg
  python auto_split.py scan.jpg -o ./output --method color --min-size 200
  python auto_split.py scan.jpg --show
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
        description="自动分割扫描图片中的多张照片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="输入扫描图片路径")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出目录（默认：与输入文件同目录下的 split_output/）",
    )
    parser.add_argument(
        "--method",
        choices=["edge", "color", "variance", "otsu", "adaptive"],
        default="edge",
        help="分割方法：edge=边缘检测，color=颜色聚类，variance=方差分析，otsu=自适应阈值（适合无白边的紧密排列扫描件），adaptive=自适应融合",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=100,
        help="最小照片尺寸（宽和高均须 ≥ 此像素值，低于此值的区域将被忽略）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="边缘检测灵敏度（0~1，越低越敏感；对应 Canny 的 low_threshold 比例）",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=10,
        help="两张照片之间的最小间隔（像素），用于合并粘连区域",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="在屏幕上显示检测结果（需 GUI 支持，Windows 可能需安装普通 opencv-python）",
    )
    parser.add_argument("--suffix", default="photo", help="Output filename prefix, e.g. photo_001.jpg")
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively process all image files in subdirectories",
    )
    return parser.parse_args()


def load_image(path: str):
    """读取图片，支持 JPG/PNG/BMP/TIFF，返回 BGR numpy 数组。
    使用 PIL 读取以支持中文路径，再转为 OpenCV 格式。"""
    try:
        pil_img = Image.open(str(path))
        pil_img.load()  # 强制加载，失败时抛异常
        arr = np.array(pil_img)
        if len(arr.shape) == 2:
            # 灰度图转为 RGB
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            # RGBA → RGB
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        else:
            # RGB → BGR（OpenCV 格式）
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    except Exception as e:
        raise ValueError(f"无法读取图片：{path}，请确认文件格式是否正确。（{e}）")


def _correct_rotation(pil_img: Image.Image, adjust_angle: bool = True) -> Image.Image:
    """检测并矫正照片的旋转角度，并裁剪白边。
    适用于照片在扫描纸上放置倾斜的情况。
    adjust_angle: True=自动矫正，False=仅返回角度信息
    """
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    if h < 100 or w < 100:
        return pil_img

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.uint8)

    # 创建非白色区域掩码（照片内容）
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找最大的非白区域（照片主体）
    largest_cnt = None
    largest_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area and area > 5000:
            largest_area = area
            largest_cnt = cnt

    if largest_cnt is None or largest_area < 10000:
        return pil_img

    # 计算最小外接旋转矩形
    rect = cv2.minAreaRect(largest_cnt)
    (cx, cy), (bw, bh), angle = rect

    # 标准化角度到 [-45, 45] 范围
    original_angle = angle
    if angle < -45:
        angle += 90
        bw, bh = bh, bw  # 交换宽高

    result = arr.copy()
    rotated_flag = False

    # 判断是否需要旋转矫正
    if abs(angle) > 3 and adjust_angle:
        rotated_flag = True
        # 旋转图像以矫正角度
        center = (cx, cy)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 计算旋转后的边界
        cos = np.abs(rot_matrix[0, 0])
        sin = np.abs(rot_matrix[0, 1])
        new_w = int(bh * sin + bw * cos)
        new_h = int(bh * cos + bw * sin)

        # 调整旋转矩阵以包含整个图像
        rot_matrix[0, 2] += (new_w - w) / 2
        rot_matrix[1, 2] += (new_h - h) / 2

        # 执行旋转
        result = cv2.warpAffine(arr, rot_matrix, (new_w, new_h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)

    # 裁剪白边：找到照片内容的实际边界
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    # 方法：从四个方向扫描，找到第一个非白像素的位置
    # 顶部
    top_y = 0
    for y in range(result_gray.shape[0]):
        if np.mean(result_gray[y, :] > 230) < 0.5:  # 如果这行超过50%非白
            top_y = y
            break

    # 底部
    bottom_y = result_gray.shape[0]
    for y in range(result_gray.shape[0] - 1, -1, -1):
        if np.mean(result_gray[y, :] > 230) < 0.5:
            bottom_y = y
            break

    # 左侧
    left_x = 0
    for x in range(result_gray.shape[1]):
        if np.mean(result_gray[:, x] > 230) < 0.5:
            left_x = x
            break

    # 右侧
    right_x = result_gray.shape[1]
    for x in range(result_gray.shape[1] - 1, -1, -1):
        if np.mean(result_gray[:, x] > 230) < 0.5:
            right_x = x
            break

    # 确保裁剪区域足够大
    crop_h = bottom_y - top_y
    crop_w = right_x - left_x
    if crop_h > 50 and crop_w > 50 and crop_h * crop_w > result.shape[0] * result.shape[1] * 0.1:
        result = result[top_y:bottom_y, left_x:right_x]

    return Image.fromarray(result)


def _fix_orientation(pil_img: Image.Image) -> Image.Image:
    """多特征融合旋转检测：饱和度偏斜 + 亮度偏斜 + 文字结构分析。
    适用于白边不明显的扫描件，尤其对含人物/建筑/文字的室内照片效果显著。
    """
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    if h < 100:
        return pil_img

    # ── 特征1：饱和度偏斜（最关键）──
    # 倒置时天花板（低饱和）在底部，地面（高饱和）在顶部 → sat_skew < 0
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    top_s = np.mean(s[:h // 2])
    bot_s = np.mean(s[h // 2:])
    sat_skew = (bot_s - top_s) / (top_s + bot_s + 1)
    sat_signal = -sat_skew  # > 0 = 倒置信号

    # ── 特征2：亮度偏斜 ──
    top_b = np.mean(arr[:h // 2, :, :])
    bot_b = np.mean(arr[h // 2:, :, :])
    bright_skew = (top_b - bot_b) / (top_b + bot_b + 1)
    bright_signal = -bright_skew  # > 0 = 倒置信号（底部更亮=天花板朝下）

    # ── 特征3：文字结构分析 ──
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    dilated = cv2.dilate(binary, kernel_h, iterations=1)
    eroded = cv2.erode(dilated, kernel_h, iterations=1)
    contours_h, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    top_lines = sum(1 for c in contours_h
                    if c.shape[0] > 10 and np.mean(c[:, 0, 1]) < h // 3)
    bot_lines = sum(1 for c in contours_h
                    if c.shape[0] > 10 and np.mean(c[:, 0, 1]) >= 2 * h // 3)
    text_skew = (top_lines - bot_lines) / (top_lines + bot_lines + 1)
    total_text = top_lines + bot_lines
    text_signal = -text_skew  # > 0 = 更多文字在底部 = 可能倒置

    # 综合决策
    # 关键规律：
    #   - sat_skew < -0.05（底部饱和度明显更低）= 天花板朝下 → 强烈倒置信号
    #   - bright_skew < -0.05（底部明显更亮）= 可能是天花板朝下
    #   - text_skew 负 + 高文本密度 = 文字结构在底部，可能倒置
    is_upside_down = False

    # 综合评分（用于条件4判断）
    confidence = sat_signal * 0.5 + bright_signal * 0.3 + text_signal * 0.2

    # 条件1: 强饱和度倒置信号
    if sat_signal > 0.05:
        is_upside_down = True

    # 条件2: 高文本密度 + 中等饱和度信号（会议/室内照片）
    elif total_text >= 5 and sat_signal > 0.02:
        is_upside_down = True

    # 条件3: 强文本信号
    elif total_text >= 3 and text_signal > 0.3:
        is_upside_down = True

    # 条件4: 综合评分足够高
    elif confidence > 0.2:
        is_upside_down = True

    if is_upside_down:
        return pil_img.rotate(180, expand=False)
    return pil_img


def save_images(boxes: list, img: np.ndarray, output_dir: Path, suffix: str, start_index: int = 1):
    """将检测到的边界框裁剪并保存到输出目录。
    使用 PIL 保存以支持中文路径，并自动修正倒置照片和旋转角度。
    start_index: 起始编号（用于批量顺序编号）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, (x, y, bw, bh) in enumerate(boxes, start=start_index):
        crop = img[y:y + bh, x:x + bw]
        # BGR → RGB 再转为 PIL Image
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_crop)
        # 自动矫正旋转角度
        pil_img = _correct_rotation(pil_img)
        # 自动修正方向
        pil_img = _fix_orientation(pil_img)
        out_path = output_dir / f"{suffix}_{i:03d}.jpg"
        pil_img.save(str(out_path), "JPEG", quality=95)
        saved.append(out_path)
        print(f"  [OK] {out_path.name}  ({bw}x{bh})")
    return saved


def draw_overlay(img: np.ndarray, boxes: list, color=(0, 255, 0)):
    """在图片上绘制矩形框（用于 --show 预览）。"""
    overlay = img.copy()
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
        # 在左上角标注序号
        cv2.putText(
            overlay, str(i), (x + 5, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
    return overlay


# ─────────────────────────────────────────────
# 方法一：边缘检测
# ─────────────────────────────────────────────
def detect_by_edge(img: np.ndarray, threshold: float = 0.5, min_size: int = 100, gap: int = 10):
    """
    通过边缘检测和轮廓查找分割照片。
    返回 sorted boxes: [(x, y, w, h), ...]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊降噪
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 边缘检测
    low = int(200 * threshold)
    high = int(low * 2.5)
    edges = cv2.Canny(blur, low, high)

    # 形态学操作：闭合小缝隙，连接断裂边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 膨胀以连接相邻区域
    dilated = cv2.dilate(closed, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    area_threshold = min_size * min_size
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤过于细长的区域（可能是文字行而非照片）
        aspect = w / max(h, 1)
        if aspect < 0.3 or aspect > 3.5:
            continue
        boxes.append((x, y, w, h))

    # 合并过于靠近的区域（gap 控制）
    boxes = merge_close_boxes(boxes, gap)

    # 按面积从大到小排序（大的优先，通常是整张扫描图；小的保留作为子照片）
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

    # 如果最大的框几乎等于整张图，说明检测的是外框，取次大的几个
    full_area = img.shape[1] * img.shape[0]
    if boxes and boxes[0][2] * boxes[0][3] > full_area * 0.8:
        boxes = boxes[1:]  # 去掉最大的外框

    return boxes


def merge_close_boxes(boxes: list, gap: int = 10) -> list:
    """合并显著重叠的区域，但不合并仅相邻/共边的独立照片。"""
    if len(boxes) <= 1:
        return list(boxes)
    merged = []
    # 按左上角位置排序（先按 y 行，再按 x 列）
    boxes_sorted = sorted(boxes, key=lambda b: (b[1] // gap * gap, b[0]))
    current = list(boxes_sorted[0])
    for nxt in boxes_sorted[1:]:
        cx1, cy1, cw1, ch1 = current
        nx, ny, nw, nh = nxt
        overlap_x = min(cx1 + cw1, nx + nw) - max(cx1, nx)
        overlap_y = min(cy1 + ch1, ny + nh) - max(cy1, ny)
        if overlap_x <= 0 or overlap_y <= 0:
            # 无重叠，直接保留当前框
            merged.append(current)
            current = list(nxt)
            continue
        # 重叠面积占较小框的比例
        ov_area = overlap_x * overlap_y
        min_area = min(cw1 * ch1, nw * nh)
        if ov_area > 0.1 * min_area:
            # 重叠超过 10%，合并
            current = [
                min(cx1, nx),
                min(cy1, ny),
                max(cx1 + cw1, nx + nw) - min(cx1, nx),
                max(cy1 + ch1, ny + nh) - min(cy1, ny),
            ]
        else:
            # 重叠不足，保留当前框，开始新框
            merged.append(current)
            current = list(nxt)
    merged.append(current)
    return merged


# ─────────────────────────────────────────────
# 方法二：颜色聚类
# ─────────────────────────────────────────────
def detect_by_color(img: np.ndarray, min_size: int = 100, gap: int = 10):
    """
    通过颜色空间转换 + 阈值分割来检测照片区域。
    假设照片区域与背景色有明显差异。
    返回 sorted boxes: [(x, y, w, h), ...]
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 方法：检测低饱和度区域（背景/纸张）的补集作为照片区域
    # 照片通常比背景更有色彩或更暗/更亮
    # 这里用自适应阈值检测亮度差异较大的区域
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊后自适应阈值
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=11, C=2,
    )

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=3)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    area_threshold = min_size * min_size
    full_area = img.shape[1] * img.shape[0]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # 排除接近整图的区域
        if w * h > full_area * 0.9:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.2 or aspect > 5.0:
            continue
        boxes.append((x, y, w, h))

    boxes = merge_close_boxes(boxes, gap)
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


# ─────────────────────────────────────────────
# 方法三：方差+饱和度分析（检测无白边的密集排布扫描件）
# ─────────────────────────────────────────────
def detect_by_otsu(img: np.ndarray, min_size: int = 100, gap: int = 10) -> list:
    """
    使用 Otsu 自适应阈值分割，适合照片之间无白边、直接贴在卡纸上的扫描件。
    通过亮度二值化 + 形态学操作提取照片区域。
    返回 sorted boxes: [(x, y, w, h), ...]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu 阈值（自动寻找最佳分割阈值）
    # 不做闭合操作，避免将相邻照片合并成一个整体
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 轻微开运算去除噪声点，但不闭合照片间的缝隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    full_area = img.shape[1] * img.shape[0]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_size * min_size:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < min_size or bh < min_size:
            continue
        # 排除接近整图的区域（可能是整个扫描页背景）
        if bw * bh > full_area * 0.85:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.2 or aspect > 5.0:
            continue
        boxes.append((x, y, bw, bh))

    boxes = merge_close_boxes(boxes, gap)

    # NMS 去重：移除高度重叠的框，保留面积最大的
    def box_iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2]); y2 = min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = b1[2] * b1[3] + b2[2] * b2[3] - inter
        return inter / union if union > 0 else 0

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    nms_result = []
    for b in boxes:
        dominated = False
        for kept in nms_result:
            if box_iou(b, kept) > 0.3:
                dominated = True
                break
        if not dominated:
            nms_result.append(b)
    boxes = nms_result

    # 按行排序（每行约 700px 高，左右并排的照片在同一行）
    boxes.sort(key=lambda b: (b[1] // 700 * 700, b[0]))
    return boxes


# ─────────────────────────────────────────────
# 方法三：自适应融合
# ─────────────────────────────────────────────
def detect_adaptive(img: np.ndarray, threshold: float, min_size: int, gap: int):
    """
    依次尝试 otsu → edge → color，综合评估后选择最佳方法。
    如果检测到图像在垂直中轴有分隔线，先拆分左右半图再分别检测。
    """
    h, w = img.shape[:2]
    full_area = w * h

    # Step 1: 检测垂直分隔线
    # 在中心区域寻找最白的列，如果明显比周围白则认为是分隔线
    center_col = w // 2
    is_split = False
    # 检查中心 ±30 列范围内的白色比例
    center_region_white = []
    for x in range(max(0, center_col - 30), min(w, center_col + 31)):
        col = img[:, x, :]
        white_ratio = np.mean(
            (col[:, 0] > 220) & (col[:, 1] > 220) & (col[:, 2] > 220)
        )
        center_region_white.append((x, white_ratio))

    # 找中心区域最白的列
    best_x, best_white = max(center_region_white, key=lambda x: x[1])
    # 同时检查两侧边缘的亮度差异
    left_edge_white = np.mean(
        np.mean(img[:, :20, :], axis=(0, 1)) > 220
    )
    right_edge_white = np.mean(
        np.mean(img[:, -20:, :], axis=(0, 1)) > 220
    )

    # 如果中心有明显白线，或中心列明显比两侧白得多，认为是分隔图
    if best_white > 0.3 and best_white > max(left_edge_white, right_edge_white) * 1.5:
        is_split = True
        print(f"  检测到垂直分隔线 (x={best_x}, white={best_white*100:.0f}%)，拆分左右半图分别检测...")

    if is_split:
        print(f"  检测到垂直分隔线，拆分左右半图分别检测...")
        mid = w // 2
        left_img = img[:, :mid, :]
        right_img = img[:, mid:, :]
        left_boxes = detect_adaptive(left_img, threshold, min_size, gap)
        right_boxes = [(x + mid, y, bw, bh) for x, y, bw, bh in
                       detect_adaptive(right_img, threshold, min_size, gap)]
        all_boxes = left_boxes + right_boxes
        print(f"  左半图: {len(left_boxes)} 张, 右半图: {len(right_boxes)} 张, 合并: {len(all_boxes)} 张")
        return all_boxes

    candidates = []

    def score_boxes(name: str, boxes: list) -> tuple:
        """返回 (score, boxes)，score 越高越好。
        先过滤掉小于 (min_size*8)^2 的碎片，再进行评分。"""
        if not boxes:
            return (0, boxes)

        # 过滤小碎片：宽和高都至少是 min_size 的 8 倍，同时面积足够大
        min_dim = min_size * 8
        min_area = (min_size * 5) ** 2
        filtered = [b for b in boxes if b[2] >= min_dim and b[3] >= min_dim and b[2] * b[3] >= min_area]
        if not filtered:
            filtered = boxes

        avg_area = sum(b[2] * b[3] for b in filtered) / len(filtered)
        max_area = max(b[2] * b[3] for b in filtered)
        num_photos = len(filtered)

        # 合理性约束：最大照片不应超过整图的 70%
        if max_area > full_area * 0.7:
            return (0, boxes)  # 不合格，可能是整页背景

        # 数量惩罚：过多碎片说明有噪声
        photo_penalty = 0
        if num_photos > 20:
            photo_penalty = -(num_photos - 20) * 10

        # 面积奖励：中等大小的照片更可信
        area_score = avg_area / full_area * 100

        # 数量分：2-8 张照片是最理想的情况
        count_score = max(0, 10 - abs(num_photos - 4) * 2)

        # 重要：如果最大照片占整图比例过大，说明可能是背景而非真实照片，大幅扣分
        max_ratio = max_area / full_area
        if max_ratio > 0.3:
            area_score *= (1.0 - max_ratio)  # 最大照片占比越大，分数越低

        total_score = area_score + count_score + photo_penalty
        return (total_score, filtered)

    # Otsu（适合无白边紧密排列）
    otsu_boxes = detect_by_otsu(img, min_size, gap)
    if otsu_boxes:
        s, b = score_boxes("otsu", otsu_boxes)
        candidates.append(("otsu", s, b))
        print(f"  Otsu: {len(otsu_boxes)} raw -> {len(b)} valid (score={s:.1f})")

    # Edge（适合有清晰边缘的扫描件）
    edge_boxes = detect_by_edge(img, threshold, min_size, gap)
    if edge_boxes:
        s, b = score_boxes("edge", edge_boxes)
        candidates.append(("edge", s, b))
        print(f"  Edge: {len(edge_boxes)} raw -> {len(b)} valid (score={s:.1f})")

    # Color（适合颜色对比明显的场景）
    color_boxes = detect_by_color(img, min_size, gap)
    if color_boxes:
        s, b = score_boxes("color", color_boxes)
        candidates.append(("color", s, b))
        print(f"  Color: {len(color_boxes)} raw -> {len(b)} valid (score={s:.1f})")

    if not candidates:
        return []

    best = max(candidates, key=lambda c: c[1])
    if best[1] > 0:
        print(f"  Selected: {best[0]} ({len(best[2])} photos)")
        return best[2]

    # 都失败了，返回数量最多的原始结果
    fallback = max(candidates, key=lambda c: len(c[2]))
    print(f"  All methods scored poorly, using {fallback[0]}")
    return fallback[2]


def deduplicate_boxes(boxes: list, iou_threshold: float = 0.5) -> list:
    """去除 IoU 超过阈值的重复框，保留面积最大的。"""
    if not boxes:
        return []

    def iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2])
        y2 = min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = b1[2] * b1[3]
        area2 = b2[2] * b2[3]
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes:
        duplicate = False
        for existing in kept:
            if iou(box, existing) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(path: Path, recursive: bool = False) -> list[Path]:
    """Collect all image files from a path (file or directory)."""
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            return [path]
        print(f"  Skip {path.name}: unsupported format '{path.suffix}'")
        return []
    elif path.is_dir():
        pattern = "**/*" if recursive else "*"
        files = sorted(path.glob(pattern))
        images = [f for f in files if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        return images
    else:
        return []


def process_one(img_path: Path, args, global_index: int, flat_output: bool = False) -> tuple[list[Path], int]:
    """Process a single image. Returns (saved_paths, next_global_index).
    flat_output=True: 所有照片输出到同一目录，全局顺序编号
    flat_output=False: 每张图片输出到独立子目录
    """
    img = load_image(img_path)
    h, w = img.shape[:2]
    stem = img_path.stem

    # 确定输出目录
    if args.output:
        if flat_output:
            # 批量模式：所有照片输出到同一目录
            output_dir = Path(args.output)
        else:
            output_dir = Path(args.output) / stem
    else:
        output_dir = img_path.parent / "split_output" / stem

    print(f"\n[{global_index:03d}] {img_path.name}")
    print(f"  Size: {w}x{h}")
    print(f"  Output: {output_dir}")

    # Detect
    if args.method == "edge":
        boxes = detect_by_edge(img, args.threshold, args.min_size, args.gap)
    elif args.method == "color":
        boxes = detect_by_color(img, args.min_size, args.gap)
    elif args.method == "otsu":
        boxes = detect_by_otsu(img, args.min_size, args.gap)
    elif args.method == "variance":
        boxes = detect_by_variance(img, args.min_size, args.gap)
    elif args.method == "adaptive":
        boxes = detect_adaptive(img, args.threshold, args.min_size, args.gap)
    else:
        print(f"  Unknown method: {args.method}")
        return [], global_index + 1

    if not boxes:
        print("  No photos detected.")
        print("  Hint: try --method color or --method adaptive, lower --threshold, reduce --min-size")
        return [], global_index + 1

    boxes.sort(key=lambda b: (b[1] // 50 * 50, b[0]))
    print(f"  Detected {len(boxes)} photo(s):")
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        print(f"    [{i:03d}] pos({x},{y}) size {bw}x{bh}")

    saved = save_images(boxes, img, output_dir, args.suffix, global_index)
    return saved, global_index + len(boxes)


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: path not found: '{input_path}'")
        sys.exit(1)

    images = collect_images(input_path, args.recursive)
    if not images:
        print(f"No image files found at: {input_path}")
        sys.exit(1)

    print(f"Found {len(images)} image(s) to process")
    print(f"Method: {args.method}")
    print("-" * 40)

    all_saved: list[Path] = []
    idx = 1
    # 批量处理时自动使用扁平输出（所有照片输出到同一目录）
    flat_output = len(images) > 1 and args.output is not None
    for img_path in images:
        saved, idx = process_one(img_path, args, idx, flat_output=flat_output)
        all_saved.extend(saved)

    print(f"\n[Done] Total {len(all_saved)} photo(s) saved across {len(images)} scan(s).")
    print(f"Output: {all_saved[0].parent if all_saved else 'N/A'}")


if __name__ == "__main__":
    main()
