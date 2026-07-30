from ultralytics import YOLO

# 加载你训练好的最佳模型
model = YOLO('runs/drone_detection/exp_ir_v_combined/weights/best.pt')
# 对单张图片进行预测
results = model(r'Video_V\val\images\V_AIRPLANE_001_frame000000.jpg', save=True)  # save=True 会保存带标注的图片
