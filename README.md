# Drone Detection Training Pipeline

基于 [Drone-detection-dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset) 数据集，使用 YOLOv8 训练无人机（及飞机、鸟类、直升机）检测模型的完整流水线。

## 功能概述

本项目提供了一套端到端的无人机目标检测训练流水线，主要功能包括：

- **视频抽帧与标注转换**：从红外（IR）和可见光（RGB）视频中按帧间隔抽取图像，并将 MATLAB `.mat` 格式的标注框自动转换为 YOLO 格式
- **数据集划分**：按 80/20 比例随机划分训练集和验证集，支持多模态（红外+可见光）数据
- **YOLOv8 模型训练**：基于 Ultralytics YOLOv8 预训练权重进行迁移学习，支持早停、自动保存最佳模型
- **多类别检测**：支持四类目标 — 飞机（AIRPLANE）、鸟类（BIRD）、无人机（DRONE）、直升机（HELICOPTER）
- **双模态感知**：同时利用红外热成像和可见光视频，提高对不同光照和天气条件下无人机的检测鲁棒性
- **模型推理**：支持对单张图片或视频文件进行目标检测，自动输出带标注框的结果

## 项目结构

```
├── configs/
│   └── data.yaml              # 数据集配置文件（路径、类别定义）
├── scripts/
│   ├── 01_extract_frames.py   # 步骤1：从视频抽帧并生成 YOLO 标签
│   ├── 02_split_dataset.py    # 步骤2：划分训练集/验证集
│   ├── 03_train.py            # 步骤3：训练 YOLOv8 模型
│   ├── 04_detect_image.py     # 步骤4：单张图片推理
│   └── 05_detect_video.py     # 步骤5：视频推理
├── weights/
│   └── yolov8n.pt             # YOLOv8 nano 预训练权重
├── requirements.txt           # Python 依赖
├── LICENSE                    # 许可证
└── README.md
```

## 流水线

```
原始视频(.mp4/.avi) + .mat标签
        │
        ▼
  [01_extract_frames.py]     ← 每5帧抽一张，bbox转为YOLO归一化格式
        │
        ▼
  YOLO_dataset/
  ├── Video_IR/  (红外图像+标签)
  └── Video_V/   (可见光图像+标签)
        │
        ▼
  [02_split_dataset.py]       ← 80%训练 / 20%验证，按模态分别划分
        │
        ▼
  YOLO_dataset/split/
  ├── Video_IR/{train,val}/{images,labels}/
  └── Video_V/{train,val}/{images,labels}/
        │
        ▼
  [03_train.py]               ← YOLOv8n 迁移学习，50 epochs
        │
        ▼
  runs/drone_detection/exp_*/weights/best.pt
        │
        ▼
  [04_detect_image.py]        ← 单图推理
  [05_detect_video.py]        ← 视频推理
```

## 依赖环境

- Python 3.8+
- CUDA（可选，GPU 训练推荐）

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包：

| 包名 | 用途 |
|------|------|
| `ultralytics` | YOLOv8 模型训练与推理 |
| `opencv-python` | 视频读取与图像处理 |
| `numpy` | 数值计算 |
| `tqdm` | 进度条显示 |
| `mcos-decoder` | 解析 .mat 格式标注文件 |

## 快速开始

### 1. 准备数据集

从 [Drone-detection-dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset) 下载数据集，将 `Data/` 文件夹放在项目根目录。数据集应包含以下结构：

```
Data/
├── Video_IR/     ← 红外视频及 .mat 标注文件
└── Video_V/      ← 可见光视频及 .mat 标注文件
```

> **注意**：视频命名规则需为 `[IR/V]_[类别]_[编号].mp4`（如 `IR_DRONE_001.mp4`），类别部分将用于自动提取标签类别。

### 2. 视频抽帧与标签转换

```bash
python scripts/01_extract_frames.py
```

- 默认每 5 帧抽取一张图像
- 自动将 bbox 从像素坐标转换为 YOLO 归一化格式
- 输出目录：`YOLO_dataset/Video_IR/` 和 `YOLO_dataset/Video_V/`

### 3. 划分训练集和验证集

```bash
python scripts/02_split_dataset.py
```

- 默认 80% 训练、20% 验证，随机种子固定为 42 保证可复现
- 输出目录：`YOLO_dataset/split/`

### 4. 训练模型

```bash
python scripts/03_train.py
```

训练配置：

| 参数 | 值 | 说明 |
|------|------|------|
| 模型 | YOLOv8n | 轻量级 nano 版本 |
| 输入尺寸 | 640×640 | 标准 YOLO 输入 |
| Epochs | 50 | 训练轮数 |
| Batch size | 16 | 批大小 |
| 设备 | GPU (device=0) | 可按需改为 `'cpu'` |
| 早停 | patience=20 | 20 轮无提升则停止 |
| 数据 | IR + 可见光 | 双模态联合训练 |

训练结果保存在 `runs/drone_detection/exp_ir_v_combined/`，包括模型权重、训练曲线和验证指标。

### 5. 推理测试

**单张图片推理：**

```bash
python scripts/04_detect_image.py
```

默认对 `Video_V/val/images/V_AIRPLANE_001_frame000000.jpg` 进行检测，结果保存至 `runs/detect/`。

**视频推理：**

```bash
python scripts/05_detect_video.py
```

默认对 `Data/Video_V/V_BIRD_017.mp4` 进行检测，输出带标注框的视频。

> 使用前请根据实际情况修改脚本中的模型路径和输入文件路径。

## 配置说明

`configs/data.yaml` 定义了数据集路径和类别信息：

```yaml
path: ./YOLO_dataset/split     # 数据集根目录
train:                          # 训练集路径
  - Video_IR/train/images
  - Video_V/train/images
val:                            # 验证集路径
  - Video_IR/val/images
  - Video_V/val/images
nc: 4                           # 类别数量
names:                          # 类别名称
  - AIRPLANE
  - BIRD
  - DRONE
  - HELICOPTER
```

## 引用

如果使用本代码，请引用原数据集论文：

- Svanström F. et al. (2020). *Real-Time Drone Detection and Tracking With Visible, Thermal and Acoustic Sensors*. ICPR 2020.

## License

本项目采用 MIT License，详见 [LICENSE](LICENSE) 文件。
