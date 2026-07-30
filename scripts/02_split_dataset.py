# scripts/02_split_dataset.py
import os
import random
import shutil
from pathlib import Path

def split_dataset(img_dir, label_dir, output_dir, train_ratio=0.8, seed=42):
    """
    将数据集拆分为训练集和验证集
    
    Args:
        img_dir: 图像文件夹路径 (如 YOLO_dataset/Video_IR/images)
        label_dir: 标签文件夹路径 (如 YOLO_dataset/Video_IR/labels)
        output_dir: 输出根目录 (如 YOLO_dataset/split)
        train_ratio: 训练集比例
        seed: 随机种子，保证可复现
    """
    random.seed(seed)
    
    # 获取所有图像文件
    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print(f"警告: {img_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(images)} 张图像")
    
    # 随机打乱
    random.shuffle(images)
    
    # 划分
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"训练集: {len(train_images)} 张")
    print(f"验证集: {len(val_images)} 张")
    
    # 创建输出目录并复制文件
    for split_name, img_list in [('train', train_images), ('val', val_images)]:
        # 创建目录
        out_img_dir = Path(output_dir) / split_name / 'images'
        out_label_dir = Path(output_dir) / split_name / 'labels'
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        
        for img_name in img_list:
            # 复制图像
            src_img = Path(img_dir) / img_name
            dst_img = out_img_dir / img_name
            shutil.copy(src_img, dst_img)
            
            # 复制对应的标签文件
            label_name = img_name.rsplit('.', 1)[0] + '.txt'
            src_label = Path(label_dir) / label_name
            dst_label = out_label_dir / label_name
            
            if src_label.exists():
                shutil.copy(src_label, dst_label)
            else:
                print(f"警告: 标签文件不存在 {src_label}")

def main():
    # ===== 配置参数 =====
    # 数据集根目录（根据你的实际情况调整）
    dataset_root = Path("YOLO_dataset")
    
    # 要处理的模态列表
    modalities = ['Video_IR', 'Video_V']
    
    # 输出目录
    output_root = dataset_root / 'split'
    
    # 训练集比例
    train_ratio = 0.8
    
    # ===== 执行拆分 =====
    for modality in modalities:
        print(f"\n{'='*50}")
        print(f"处理 {modality}...")
        print(f"{'='*50}")
        
        img_dir = dataset_root / modality / 'images'
        label_dir = dataset_root / modality / 'labels'
        
        if not img_dir.exists():
            print(f"跳过: {img_dir} 不存在")
            continue
        
        split_dataset(
            img_dir=img_dir,
            label_dir=label_dir,
            output_dir=output_root / modality,  # 每个模态单独输出
            train_ratio=train_ratio
        )
    
    print(f"\n✅ 拆分完成！数据集保存在: {output_root}")
    print("\n目录结构:")
    print(f"  {output_root}/")
    for modality in modalities:
        print(f"  ├── {modality}/")
        print(f"  │   ├── train/")
        print(f"  │   │   ├── images/")
        print(f"  │   │   └── labels/")
        print(f"  │   └── val/")
        print(f"  │       ├── images/")
        print(f"  │       └── labels/")

if __name__ == "__main__":
    main()