from ultralytics import YOLO, RTDETR
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
# 1) Load model
model = YOLO("weight/yolo26m.pt")  # 可换成yolov7,yolov8,yolov11等；首次运行会自动下载预训练权重
# 2) Train
train_results = model.train(
    data="/root/autodl-tmp/cholecTrack20_yolo/data.yaml",
    epochs=30,
    imgsz=960,
    device=0,
    batch=8,
    project="runs_cholec",     # 推荐：固定输出目录
    name="rtdetr_l_640",        # 推荐：实验名
    cache=False,               # 可选：数据太大就别 cache
    amp=True,
    patience=10,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.3,
    close_mosaic=30,
    dropout=0.3,
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.01,
    weight_decay=0.0005,
)

# 3) Validate (显式指定 data 更稳)
metrics = model.val(
    data="/root/autodl-tmp/cholecTrack20_yolo/data.yaml",
    device=0,
    imgsz=960,
    batch=8,
    plots=True,
    save_json=True,
)
# 4) Predict (server 环境建议 save=True)
results = model.predict(
    source="../yolov26/11.png",  # 注意路径
    device=0,
    save=True,
    conf=0.25
)
# 5) Export
path = model.export(format="onnx")
print("Exported to:", path)
