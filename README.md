# surgeWM-yolov26
Based on the cholecTrack20 dataset, the YOLOv26 model is used to detect sudden instrument changes in the generated surgical videos.
# 数据集
YOLO26 训练部分使用 CholecTrack20 数据集，用于手术器械检测与追踪任务。
# 项目结构
```bash
convert_cholecTrack20_to_yolo_track.py #把数据集转换为yolo训练的格式
yolo.py #对yolo26训练
track.py #用已训练好的yolo模型对目标视频进行追踪，并保存每条轨迹的信息
count_instrument_changes.py #检测被测视频出现多少次器械突变
# 训练
cd yolov26
python yolo.py
## Installation
```bash
# Clone the repository
git clone https://github.com/vamos-jpg/surgeWM-yolov26
# 安装yolo所需依赖
pip install ultralytics
python -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"
# 训练效果
