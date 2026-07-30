from ultralytics import YOLO

# 加载你训练好的最佳模型
model = YOLO('runs/drone_detection/exp_ir_v_combined/weights/best.pt')

results = model(r'\Data\Video_V\V_BIRD_017.mp4', save=True)