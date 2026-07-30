# train_drone_detector.py
from ultralytics import YOLO
import os

def main():
    # 检查权重是否存在
    weight_path = 'weights/yolov8n.pt'
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件不存在: {weight_path}")
        print("请手动下载 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt")
        return
    
    # 加载模型
    print("✅ 加载模型...")
    model = YOLO(weight_path)
    
    # 训练参数
    # /*
    # results = model.train(
    #     data='configs/data.yaml',
    #     epochs=50,
    #     imgsz=640,
    #     batch=16,
    #     device= 0, #'cpu',  # 有GPU时改为0或cuda设备号
    #     workers=4,
    #     patience=20,  # 早停
    #     save=True,
    #     project='runs/drone_detection',
    #     name='exp1'
    # )*/
    results = model.train(
    data='configs/data.yaml',  # 使用更新后的配置
    epochs=50,
    imgsz=640,
    batch=16,  # 注意：数据量翻倍，可能需要调整
    device=0,
    workers=8,
    patience=20,
    save=True,
    project='runs/drone_detection',
    name='exp_ir_v_combined'
    )
    
    print("✅ 训练完成！")
    
    # 验证
    print("\n📊 验证模型...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()