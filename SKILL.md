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
# 单张图片（在图片所在目录创建 [图片名]_output/）
python {baseDir}/scripts/auto_split.py <输入图片路径>

# 整个目录批量处理（在输入目录下创建 output/，所有照片保存到同一目录）
python {baseDir}/scripts/auto_split.py <图片目录路径>

# 递归处理子目录中的图片
python {baseDir}/scripts/auto_split.py <图片目录路径> -r

# 指定输出目录（批量时所有照片输出到同一目录，全局顺序编号）
python {baseDir}/scripts/auto_split.py <图片目录路径> -o ./结果

# 最小分割尺寸
python {baseDir}/scripts/auto_split.py <输入图片路径> --min-size 200

# 调整边缘检测灵敏度（默认已降低到0.3，更敏感）
python {baseDir}/scripts/auto_split.py <输入图片路径> --threshold 0.2

# 按颜色聚类分割（适合背景色统一的扫描件）
python {baseDir}/scripts/auto_split.py <输入图片路径> --method color

# Otsu 自适应阈值（适合照片无白边、直接贴在卡纸上的扫描件）
python {baseDir}/scripts/auto_split.py <输入图片路径> --method otsu

# 强制使用边缘检测（不推荐，除非特殊需求）
python {baseDir}/scripts/auto_split.py <输入图片路径> --method edge
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `<input>` | 输入扫描图片路径 **或目录路径**（支持 JPG/PNG/BMP/TIFF） | 必填 |
| `-o, --output` | 输出目录 | 默认：单张图片创建 [文件名]_output/，目录处理创建 output/ |
| `-r, --recursive` | 递归处理子目录中的所有图片 | 关闭 |
| `--method` | 分割方法：`edge`（边缘检测）/ `color`（颜色聚类）/ `otsu`（自适应阈值，适合无白边紧密排列）/ `adaptive`（自动选择，推荐） | `adaptive` |
| `--min-size` | 最小照片尺寸（宽和高均须 ≥ 此值） | 100 像素 |
| `--threshold` | 边缘检测灵敏度（越低越敏感） | 0.3 |
| `--gap` | 两张照片之间的最小间隔（像素），用于合并粘连区域 | 10 |
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

### 方法三：自适应（自动选择最优方法，推荐）
1. 依次尝试 Otsu、Edge、Color 三种方法
2. 对每种方法的检测结果进行评分
3. 选择得分最高的方法的结果
4. 适合各种类型的扫描件，是最可靠的方法

## 自动方向校正

分割完成后，脚本会对每张照片进行**智能旋转检测**，自动修正倒置的照片。

### 检测原理

采用多特征融合评分（无需 Tesseract OCR，纯 OpenCV + NumPy）：

| 特征 | 原理 | 权重 |
|------|------|------|
| **饱和度偏斜** | 倒置照片底部（天花板）饱和度低于顶部 | 50% |
| **亮度偏斜** | 倒置照片底部（天花板）比顶部亮 | 30% |
| **文字结构** | 检测水平线条密度，文字集中在底部暗示倒置 | 20% |

### 适用场景

- ✓ 室内会议/合影照片（天花板 vs 地面对比明显）
- ✓ 建筑照片（天空 vs 地面饱和度差异）
- ✓ 含横幅/标语/文字的照片（文字结构分析）
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

## 扫描建议

为了获得最佳的分割效果，建议按照以下方法摆放照片：

### 1. 照片摆放方法
- **尽量贴近扫描仪边缘**：将照片紧贴扫描仪的四周边缘放置，避免照片倾斜
- **保持平整**：确保照片表面平整，没有弯曲或卷边
- **适当间距**：照片之间保持适当间隔（建议5-10mm），便于算法区分

### 2. 照片朝向
- **统一方向**：所有照片的正面朝向应与扫描方向一致（都朝上或都朝下）
- **避免混放**：不要将正向和反向的照片混在同一扫描图中
- **文字朝上**：如有文字内容，确保文字朝上摆放

这样做的好处：
- 减少算法的计算负担，提高分割准确率
- 避免因照片倾斜导致的识别错误
- 简化自动方向校正的难度，减少误判

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
| 检测不到任何照片 | 尝试降低 `--threshold`（如 0.2）或换用 `--method color` |
| 一张照片被切成多块 | 增大 `--gap` 值（如 30），或增大 `--min-size` |
| 两张照片被合并成一块 | 减小 `--gap` 值（如 5），或降低 `--threshold` |
| 输出图片质量差 | 输入文件本身分辨率过低；或增加 `--min-size` 过滤小区域 |
| 图片方向错误 | 先手动旋转输入图到正确方向，再运行脚本 |
| Windows 下 --show 报错 | 安装 `pip install opencv-python-headless` 改为无 GUI 模式，或去掉 `--show` |
