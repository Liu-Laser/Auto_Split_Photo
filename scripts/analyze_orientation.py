#!/usr/bin/env python3
"""
分析扫描图中每张照片的实际方向
"""

import cv2
import numpy as np
from PIL import Image

def analyze_photo_orientation():
    # 读取原始扫描图
    img_path = 'C:/Users/liulaser/Pictures/扫描/扫描_20260820.jpg'
    pil_img = Image.open(img_path)
    arr = np.array(pil_img)
    h, w = arr.shape[:2]

    # 根据之前检测结果的位置
    photos = [
        (0, 0, 2053, 2934),      # photo_001
        (3024, 8, 1976, 2904),   # photo_002
        (3013, 4143, 2027, 2539), # photo_003
        (0, 4240, 2098, 2711)    # photo_004
    ]

    print("扫描图照片方向分析:")
    print("=" * 60)

    for i, (x, y, bw, bh) in enumerate(photos, 1):
        # 裁剪出照片区域
        crop = arr[y:y+bh, x:x+bw]

        # 检查长宽比
        aspect = bw / bh

        # 转换为BGR格式（OpenCV格式）
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # 分析边缘密度
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

        h_energy = np.mean(np.abs(sobel_h))
        v_energy = np.mean(np.abs(sobel_v))

        # 判断是否可能是横版照片
        # 如果横版，应该有更多水平边缘
        is_likely_landscape = h_energy > v_energy * 1.2

        # 检查人脸分布（如果有）
        face_score = 0
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) > 0:
                # 统计人脸在上下半部分的分布
                top_faces = sum(1 for x_f, y_f, fw, fh in faces if y_f + fh < bh // 2)
                bot_faces = len(faces) - top_faces
                if bot_faces > top_faces:
                    face_score = 0.5  # 人脸在底部，可能是正常竖版
                else:
                    face_score = -0.5  # 人脸在顶部，可能是倒置或横版
        except:
            pass

        print(f"\n照片 {i}:")
        print(f"  位置: ({x}, {y})")
        print(f"  尺寸: {bw} x {bh}")
        print(f"  宽高比: {aspect:.2f}")
        print(f"  当前方向: {'横版' if bw >= bh else '竖版'}")
        print(f"  水平边缘能量: {h_energy:.2f}")
        print(f"  垂直边缘能量: {v_energy:.2f}")
        print(f"  边缘比 (H/V): {h_energy/v_energy:.2f}")
        print(f"  可能是横版: {'是' if is_likely_landscape else '否'}")
        print(f"  人脸分布: {face_score:.2f}")

        # 保存样本以供检查
        sample_path = f'sample_photo_{i}.jpg'
        pil_crop = Image.fromarray(crop)
        pil_crop.save(sample_path)
        print(f"  样本已保存: {sample_path}")

if __name__ == "__main__":
    analyze_photo_orientation()