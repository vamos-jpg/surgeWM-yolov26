import json
from pathlib import Path
import numpy as np
import pandas as pd
df=pd.read_csv('../out/vid01/vid01_tracked_track_summary.csv')
# 筛选 class_switch_count 大于 0 的轨迹
switch_tracks = df[df['class_switch_count'] > 0]

# 统计这样的轨迹占比，即器械突变次数
num = switch_tracks.shape[0]
print(num/len(df))