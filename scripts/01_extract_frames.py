import cv2
import os
import numpy as np
from mcos_decoder import load_groundtruth
from tqdm import tqdm

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """将 (x, y, w, h) 转换为 YOLO 格式 (x_center, y_center, width, height) 归一化"""
    x, y, w, h = bbox
    x_center = (x + w/2) / img_width
    y_center = (y + h/2) / img_height
    width = w / img_width
    height = h / img_height
    # 防止数值溢出
    return max(0, min(1, x_center)), max(0, min(1, y_center)), max(0, min(1, width)), max(0, min(1, height))

def process_video(video_path, label_path, output_img_dir, output_label_dir, class_mapping, frame_interval=5):
    """
    处理单个视频：抽帧并生成YOLO标签
    - video_path: 视频文件路径
    - label_path: .mat标签文件路径
    - output_img_dir: 输出图像文件夹
    - output_label_dir: 输出标签文件夹
    - class_mapping: 类别名到ID的映射字典
    - frame_interval: 每隔多少帧抽取一张
    """
    # 读取标签
    bboxes_per_frame = load_groundtruth(label_path)  # list of (x,y,w,h) or None
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 视频名称（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔 frame_interval 帧处理一次
        if frame_idx % frame_interval == 0:
            # 检查该帧是否有标签
            bbox = bboxes_per_frame[frame_idx] if frame_idx < len(bboxes_per_frame) else None
            
            if bbox is not None:
                # 提取类别（需要从标签文件名推断，见下方说明）
                # 这里假设类别已通过文件名或外部映射确定
                # 临时示例：从视频文件名提取类别（如 IR_DRONE_001 -> class_id=2）
                class_name = extract_class_from_filename(video_name)
                if class_name not in class_mapping:
                    print(f"未知类别: {class_name}，跳过帧 {frame_idx}")
                    frame_idx += 1
                    continue
                class_id = class_mapping[class_name]
                
                # 转换bbox
                x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)
                
                # 保存图像
                img_filename = f"{video_name}_frame{frame_idx:06d}.jpg"
                img_path = os.path.join(output_img_dir, img_filename)
                cv2.imwrite(img_path, frame)
                
                # 保存标签
                label_filename = f"{video_name}_frame{frame_idx:06d}.txt"
                label_path_out = os.path.join(output_label_dir, label_filename)
                with open(label_path_out, 'w') as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"视频 {video_name} 处理完成，保存了 {saved_count} 张标注图像")

def extract_class_from_filename(filename):
    """从文件名提取类别，如 IR_DRONE_001 -> DRONE"""
    # 示例：根据你的命名规则调整
    parts = filename.split('_')
    if len(parts) >= 3:
        return parts[1]  # 取第二个部分作为类别名，格式: [IR/VIS]_[类别]_[编号]
    return None

# 使用示例
if __name__ == "__main__":
    # 类别映射（根据你的数据集实际情况）
    class_mapping = {
        'AIRPLANE': 0,
        'BIRD': 1,
        'DRONE': 2,
        'HELICOPTER': 3
    }
    
    # 路径设置
    data_root = "Data"  # 数据集根目录
    output_root = "YOLO_dataset"
    os.makedirs(output_root, exist_ok=True)
    
    # 分别处理红外和可见光视频
    for modality in ["Video_IR", "Video_V"]:
        video_dir = os.path.join(data_root, modality)
        label_dir = os.path.join(data_root, modality)  # 假设标签在同一目录
        
        # 输出目录
        img_dir = os.path.join(output_root, modality, "images")
        label_dir_out = os.path.join(output_root, modality, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir_out, exist_ok=True)
        
        # 遍历所有视频
        for video_file in os.listdir(video_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(video_dir, video_file)
                label_file = video_file.replace('.mp4', '_LABELS.mat').replace('.avi', '_LABELS.mat')
                label_path = os.path.join(label_dir, label_file)
                
                if os.path.exists(label_path):
                    process_video(
                        video_path, label_path, 
                        img_dir, label_dir_out, 
                        class_mapping, 
                        frame_interval=5  # 每5帧取1张
                    )
                else:
                    print(f"未找到标签文件: {label_path}")