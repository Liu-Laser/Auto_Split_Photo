# Auto Split Photo — 扫描照片分割工具

从一张包含多张照片的扫描图中，自动检测每张照片的边界，并将它们分割为独立的图片文件输出。

## 功能特点

- **多种检测方法**：边缘检测、颜色聚类、Otsu 自适应阈值
- **智能选择**：自动评估并选择最优检测方法
- **旋转矫正**：自动检测并矫正倾斜照片
- **白边裁剪**：智能裁剪照片周围的白边
- **方向校正**：自动检测并修正倒置照片
- **批量处理**：支持目录批量处理和顺序编号

## 安装依赖

```bash
pip install opencv-python Pillow numpy
```

或在脚本目录下运行：

```bash
python scripts/install_requirements.py
```

## 使用方式

### 单张图片
```bash
python scripts/auto_split.py <输入图片路径>
```

### 批量处理（推荐）
```bash
python scripts/auto_split.py <图片目录路径> -o ./结果 --method adaptive --min-size 50
```

### 递归处理子目录
```bash
python scripts/auto_split.py <图片目录路径> -r -o ./结果
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `<input>` | 输入扫描图片路径或目录路径 | 必填 |
| `-o, --output` | 输出目录 | 与输入文件同目录下的 `split_output/` |
| `-r, --recursive` | 递归处理子目录中的所有图片 | 关闭 |
| `--method` | 分割方法：`edge` / `color` / `otsu` / `adaptive` | `edge` |
| `--min-size` | 最小照片尺寸（宽和高均须 ≥ 此值） | 100 像素 |
| `--threshold` | 边缘检测灵敏度（越低越敏感） | 0.5 |
| `--gap` | 两张照片之间的最小间隔（像素） | 10 |

## 输出格式

批量处理时，所有照片输出到同一目录，全局顺序编号：
```
结果/
├── photo_001.jpg  (来自扫描图1)
├── photo_002.jpg  (来自扫描图1)
├── photo_003.jpg  (来自扫描图2)
└── ...
```

## 工作原理

### 1. 照片检测
- **边缘检测**：Canny 边缘 + 轮廓查找
- **颜色聚类**：HSV 色彩空间阈值分割
- **Otsu 自适应**：自动寻找最佳分割阈值（适合无白边的紧密排列）
- **自适应融合**：综合评估各方法，选择最优结果

### 2. 旋转矫正
- 检测照片内容的旋转角度（最小外接矩形）
- 自动旋转矫正（阈值 > 3°）
- 裁剪旋转产生的白边

### 3. 方向校正
- 多特征融合评分：饱和度偏斜 + 亮度偏斜 + 文字结构
- 自动检测倒置照片并旋转 180°

## 示例

```bash
# 处理单个扫描图
python scripts/auto_split.py scan.jpg --method adaptive --min-size 50

# 批量处理整个目录
python scripts/auto_split.py "C:/Pictures/scans" -o "./output" -r

# 指定输出目录
python scripts/auto_split.py "input.jpg" -o "C:/output" --method otsu
```

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| 检测不到任何照片 | 尝试降低 `--threshold` 或换用 `--method color` |
| 一张照片被切成多块 | 增大 `--gap` 值（如 30），或增大 `--min-size` |
| 两张照片被合并成一块 | 减小 `--gap` 值（如 5），或降低 `--threshold` |
| Windows 中文路径问题 | 已修复，使用 PIL 读取以支持中文路径 |

## 依赖

- Python 3.8+
- opencv-python
- Pillow
- numpy

## License

MIT License
