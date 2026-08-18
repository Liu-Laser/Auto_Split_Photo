---
name: auto_split
description: 识别扫描图片中的多张照片并分割为独立图片。触发词：识别图片、分割照片、切照片、扫图拆分、一图多张、扫描件切图。
version: 1.0.0
metadata:
  openclaw:
    requires:
      anyBins:
        - python
      pip:
        - opencv-python
        - Pillow
        - numpy
---

# Auto Split — 扫描照片分割工具

从一张包含多张照片的扫描图中，自动检测每张照片的边界，并将它们分割为独立的图片文件输出。

## Script Directory

脚本位于 `{baseDir}/scripts/` 目录下。`{baseDir}` = 本 SKILL.md 所在目录。

| 脚本 | 用途 |
|------|------|
| `scripts/auto_split.py` | 主脚本：检测边界并分割照片 |
| `scripts/install_requirements.py` | 依赖安装脚本 |

## 使用方式

```bash
# 单张图片
python {baseDir}/scripts/auto_split.py <输入图片路径>

# 整个目录批量处理（自动扫描所有图片文件）
python {baseDir}/scripts/auto_split.py <图片目录路径>

# 递归处理子目录中的图片
python {baseDir}/scripts/auto_split.py <图片目录路径> -r

# 指定输出目录（批量时所有照片输出到同一目录，全局顺序编号）
python {baseDir}/scripts/auto_split.py <图片目录路径> -o ./结果

# 最小分割尺寸
python {baseDir}/scripts/auto_split.py <输入图片路径> --min-size 200

# 调整边缘检测灵敏度
python {baseDir}/scripts/auto_split.py <输入图片路径> --threshold 0.3

# 按颜色聚类分割（适合背景色统一的扫描件）
python {baseDir}/scripts/auto_split.py <输入图片路径> --method color

# Otsu 自适应阈值（适合照片无白边、直接贴在卡纸上的扫描件）
python {baseDir}/scripts/auto_split.py <输入图片路径> --method otsu

# 自动选择最优方法（推荐，默认）
python {baseDir}/scripts/auto_split.py <输入图片路径> --method adaptive
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `<input>` | 输入扫描图片路径 **或目录路径**（支持 JPG/PNG/BMP/TIFF） | 必填 |
| `-o, --output` | 输出目录 | 与输入文件同目录下的 `split_output/` |
| `-r, --recursive` | 递归处理子目录中的所有图片 | 关闭 |
| `--method` | 分割方法：`edge`（边缘检测，默认）/ `color`（颜色聚类）/ `otsu`（自适应阈值，适合无白边紧密排列）/ `adaptive`（自动选择） | `edge` |
| `--min-size` | 最小照片尺寸（宽和高均须 ≥ 此值） | 100 像素 |
| `--threshold` | 边缘检测灵敏度（越低越敏感） | 0.5 |
| `--gap` | 两张照片之间的最小间隔（像素），防止粘连照片被误分 | 10 |
| `--suffix` | 输出文件名后缀前缀 | `photo_` |

## 工作原理

### 方法一：边缘检测（默认，推荐）
1. 将图片转为灰度并进行高斯模糊降噪
2. 使用 Canny 或自适应阈值检测边缘
3. 查找所有轮廓，过滤掉太小的噪声
4. 对轮廓进行合并（`--gap` 参数控制粘连合并）
5. 按面积排序，提取每个区域的 bounding box
6. 裁剪并保存为独立图片

### 方法二：颜色聚类
1. 将图片转为 HSV 色彩空间
2. 使用 K-Means 或均值漂移聚类背景色区域
3. 分离前景（照片内容）与背景
4. 提取各连通区域并裁剪

### 方法三：自适应（先 edge 后 color）
1. 先用边缘检测方法
2. 如果检测到的照片数量过少（≤1），改用颜色聚类重试
3. 融合两种方法的检测结果，取并集

## 自动方向校正

分割完成后，脚本会对每张照片进行**智能旋转检测**，自动修正倒置的照片。

### 检测原理

采用多特征融合评分（使用 MediaPipe Pose + Face Mesh + 传统 CV）：

| 特征 | 原理 | 权重 |
|------|------|------|
| **MediaPipe 姿态估计** | 检测肩部连线角度，判断站立方向 | 30% |
| **人脸位置分析** | 检测多个人脸在上下半部分的分布 | 30% |
| **MediaPipe 人脸关键点** | 检测脸颊骨连线角度，判断 90°/270° 旋转 | 20% |
| **亮度/饱和度差值** | 天花板 vs 地面的亮度和饱和度差异 | 10% |
| **四角白边分布** | 对角线白边差异检测倒置 | 5% |
| **文字结构方向** | 水平线条集中在顶部暗示倒置 | 5% |

### 支持的方向

- ✓ **180° 倒置**：照片头朝下、脚朝上
- ✓ **90° 旋转**：照片顺时针旋转了 90°
- ✓ **270° 旋转**：照片逆时针旋转了 270°（或顺时针 90°）

### 适用场景

- ✓ 室内合影照片（天花板 vs 地面明显）
- ✓ 建筑照片（天空 vs 地面饱和度差异）
- ✓ 含横幅/标语/文字的照片
- ✓ 单人站立照片（姿态估计可靠）
- △ 户外风景照（天空/植被饱和度差异较小，可能误判）

### 手动覆盖

```bash
# 跳过方向校正（如需手动处理）
python auto_split.py scan.jpg --no-rotate
```

### 单张图片
```
split_output/
├── photo_001.jpg
├── photo_002.jpg
└── ...
```

### 批量目录（指定 `-o` 输出目录）
所有照片输出到同一目录，**全局顺序编号**：
```
结果/
├── photo_001.jpg  (来自扫描图1)
├── photo_002.jpg  (来自扫描图1)
├── photo_003.jpg  (来自扫描图2)
├── ...
└── photo_021.jpg  (来自扫描图5)
```

### 批量目录（不指定 `-o`）
每张图片输出到独立子目录：
```
split_output/
├── scan_001/
│   ├── photo_001.jpg
│   └── photo_002.jpg
├── scan_002/
│   └── photo_001.jpg
└── ...
```

## 常见使用场景

- **相册扫描**：一张扫描图包含多张老照片，需要分别提取
- **证件扫描件**：身份证/护照扫描件上的正反面分割
- **集体照裁剪**：大合影中单人照片分割
- **票据拆分**：多张收据/发票拼在同一页的扫描图拆分

## 前置依赖

运行前确保已安装所需 Python 包：

```bash
pip install opencv-python Pillow numpy
```

或在脚本目录下运行：

```bash
python {baseDir}/scripts/install_requirements.py
```

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| 检测不到任何照片 | 尝试降低 `--threshold`（如 0.3）或换用 `--method color` |
| 一张照片被切成多块 | 增大 `--gap` 值（如 30），或增大 `--min-size` |
| 两张照片被合并成一块 | 减小 `--gap` 值（如 5），或降低 `--threshold` |
| 输出图片质量差 | 输入文件本身分辨率过低；或增加 `--min-size` 过滤小区域 |
| 图片方向错误 | 先手动旋转输入图到正确方向，再运行脚本 |
| Windows 下 --show 报错 | 安装 `pip install opencv-python-headless` 改为无 GUI 模式，或去掉 `--show` |
