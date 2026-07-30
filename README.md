# Drone Detection Training Pipeline

基于 [Drone-detection-dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset) 数据集，使用 YOLOv8 训练无人机（及飞机、鸟类、直升机）检测模型。

## 依赖环境

- Python 3.8+
- 安装依赖：`pip install -r requirements.txt`

## 数据集准备

1. 从原仓库下载数据集（或使用已有副本），将 `Data/` 文件夹放在项目根目录。
2. 运行 `python scripts/01_extract_frames.py` 从视频抽帧并生成 YOLO 格式标签。
3. 运行 `python scripts/02_split_dataset.py` 划分训练集和验证集。

## 训练模型

运行 `python scripts/03_train.py` 开始训练，结果保存在 `runs/` 中。

## 推理测试

- 单张图片：`python scripts/04_detect_image.py`
- 视频：`python scripts/05_detect_video.py`

## 引用

如果使用本代码，请引用原数据集论文：

- Svanström F. et al. (2020). Real-Time Drone Detection and Tracking With Visible, Thermal and Acoustic Sensors. ICPR 2020.