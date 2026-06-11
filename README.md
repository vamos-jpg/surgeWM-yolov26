# surgeWM-yolov26
Based on the cholecTrack20 dataset, the YOLOv26 model is used to detect sudden instrument changes in the generated surgical videos.
# CholecTrack20 YOLO Tracking Pipeline

This project converts the **CholecTrack20** dataset to YOLO format, trains detection/tracking models, and performs video tracking with detailed analysis of surgical instruments.

## Features

- Convert CholecTrack20 dataset frames and annotations to YOLO format.
- Generate class-balanced training dataset with oversampling.
- Train YOLO models (YOLOv26m or other variants) for instrument detection.
- Track instruments in videos using ByteTrack or BotSort tracker.
- Compute statistics for stable class switches (instrument changes).
- Save visualizations, frame-level CSV, track-level JSON, and summaries.

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo>

# Install dependencies
pip install -r requirements.txt

