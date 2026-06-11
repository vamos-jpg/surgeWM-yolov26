# SurgWM-yolo26

<div align="center">

**生成手术视频的器械突变检测**

基于 `YOLOv26` 进行器械检测/跟踪实验。

</div>

---

## 项目简介

本项目基于 CholecTrack20 数据集，对yolov26进行手术器械目标检测与追踪

---

## 可视化结果

### YOLO 训练结果曲线

> 展示 YOLO 训练过程中的关键指标变化，可用于观察收敛情况、精度趋势与训练稳定性。

<p align="center">
  <img src="assert/yolo_results.png" alt="YOLO Training Results" width="850">
</p>

---

## 仓库结构

```text
SurgWM-yolov26/
├── assert/                                # 可视化结果
├── convert_cholecTrack20_to_yolo_track.py # 将数据集转换为yolo训练格式
├── yolo.py                                # 训练脚本
├── track.py                               # 用训练好的模型对目标视频进行追踪检测
|—— count_instrument_changes.py            # 检测视频器械突变次数
└── README.md
```

---

## 数据集说明



### CholecTrack20

YOLO 训练部分使用 `CholecTrack20` 数据集，用于手术器械检测与跟踪任务。

---

## 环境准备

### 安装 YOLO 所需依赖

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"
```

---

## 使用说明

## YOLOv26

用于手术器械检测、训练与基础跟踪实验。

### 训练

```bash
cd yolov26
python yolo.py
```

训练配置示例：

```python
from ultralytics import YOLO

model = YOLO("weight/yolo26m.pt")

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
```

### 数据集配置示例

```yaml
path: /path/to/cholec_dataset
train: images/train
val: images/val

nc: 7
names: ['class0', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6']
```

### 推理与跟踪

```bash
python yolo.py
python track.py
```

---

## 参考项目

- Ultralytics: https://github.com/ultralytics/ultralytics

---

## 致谢

感谢 `Ultralytics` 社区提供的开源工具与基线实现，为手术视频理解、评测与可视化实验提供了良好基础。
