import os
import cv2
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict, deque, Counter
from ultralytics import YOLO, RTDETR
from tqdm import tqdm
def draw_fading_trail(overlay, points, base_color=(0, 255, 255), max_thickness=6):
    if len(points) < 2:
        return overlay
    n = len(points)
    for i in range(1, n):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        t = i / (n - 1 + 1e-6)
        alpha = 0.15 + 0.85 * t
        thickness = int(1 + (max_thickness - 1) * t)
        color = (
            int(base_color[0] * alpha),
            int(base_color[1] * alpha),
            int(base_color[2] * alpha),
        )
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return overlay
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
def mode_or_none(values):
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]
def count_stable_class_switches(class_names, frames=None, min_stable_frames=3):
    """
    统计稳定类别突变次数。

    规则：
    - 当前轨迹有一个 confirmed_class，表示当前确认类别；
    - 当检测到不同类别 candidate_class 时，先不立刻算突变；
    - 只有 candidate_class 连续出现 >= min_stable_frames 次，
      才确认从 confirmed_class 突变到 candidate_class；
    - 短暂的 1~2 帧误分类不会被计入突变。

    返回：
    - switch_count: 稳定突变次数
    - switch_events: 每次突变的详细信息
    - stable_class_names: 平滑后的类别序列
    """
    if not class_names:
        return 0, [], []

    min_stable_frames = max(1, int(min_stable_frames))

    confirmed_class = class_names[0]
    stable_class_names = [confirmed_class]

    candidate_class = None
    candidate_start_idx = None
    candidate_count = 0

    switch_count = 0
    switch_events = []

    for i in range(1, len(class_names)):
        current_class = class_names[i]

        if current_class == confirmed_class:
            # 回到已确认类别，说明之前的候选突变不稳定，清空候选
            candidate_class = None
            candidate_start_idx = None
            candidate_count = 0
            stable_class_names.append(confirmed_class)
            continue

        # current_class != confirmed_class，出现可能的新类别
        if current_class == candidate_class:
            candidate_count += 1
        else:
            candidate_class = current_class
            candidate_start_idx = i
            candidate_count = 1

        if candidate_count >= min_stable_frames:
            old_class = confirmed_class
            new_class = candidate_class

            switch_count += 1

            start_frame = frames[candidate_start_idx] if frames is not None else None
            confirm_frame = frames[i] if frames is not None else None

            switch_events.append({
                "from_class": old_class,
                "to_class": new_class,
                "start_index": int(candidate_start_idx),
                "confirm_index": int(i),
                "start_frame": int(start_frame) if start_frame is not None else None,
                "confirm_frame": int(confirm_frame) if confirm_frame is not None else None,
                "stable_frames": int(candidate_count),
            })

            # 确认突变
            confirmed_class = new_class

            # 把候选类别开始到当前帧这一段回填为新确认类别
            for k in range(candidate_start_idx, i + 1):
                if k < len(stable_class_names):
                    stable_class_names[k] = confirmed_class
                else:
                    stable_class_names.append(confirmed_class)

            candidate_class = None
            candidate_start_idx = None
            candidate_count = 0
        else:
            # 尚未达到稳定帧数，暂时仍认为是原类别
            stable_class_names.append(confirmed_class)

    return switch_count, switch_events, stable_class_names

def build_track_outputs(records, fps, video_name, model_names, min_stable_frames=3):
    """
    将逐帧 records 聚合为：
    1) tracks_json: 每条 track 的完整时序信息
    2) summary_rows: 每条 track 的摘要信息，方便保存 CSV
    """
    grouped = defaultdict(list)
    for row in records:
        grouped[int(row["track_id"])].append(row)
    tracks_json = []
    summary_rows = []
    for tid, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x["frame"])
        frames = [int(r["frame"]) for r in rows]
        boxes = [[int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])] for r in rows]
        centers = [[float(r["cx"]), float(r["cy"])] for r in rows]
        class_ids = [int(r["class_id"]) for r in rows]
        class_names = [str(r["class_name"]) for r in rows]
        confs = [float(r["conf"]) for r in rows]
        widths = [float(r["w"]) for r in rows]
        heights = [float(r["h"]) for r in rows]
        areas = [float(r["area"]) for r in rows]
        velocities = [[0.0, 0.0]]
        speeds = [0.0]
        frame_gaps = [0]
        for i in range(1, len(rows)):
            dt_frames = max(1, frames[i] - frames[i - 1])
            dt_sec = dt_frames / fps if fps > 0 else dt_frames
            vx = (centers[i][0] - centers[i - 1][0]) / dt_sec
            vy = (centers[i][1] - centers[i - 1][1]) / dt_sec
            velocities.append([float(vx), float(vy)])
            speeds.append(float(math.sqrt(vx * vx + vy * vy)))
            frame_gaps.append(int(dt_frames))
        dominant_class = mode_or_none(class_names)

        class_switch_count, class_switch_events, stable_class_names = count_stable_class_switches(
            class_names=class_names,
            frames=frames,
            min_stable_frames=min_stable_frames,
        )

        has_class_mutation = class_switch_count > 0

        unique_classes = sorted(set(class_names))
        stable_unique_classes = sorted(set(stable_class_names))

        total_distance = sum(euclidean(centers[i], centers[i - 1]) for i in range(1, len(centers)))
        duration_sec = (frames[-1] - frames[0] + 1) / fps if fps > 0 else len(frames)
        track_obj = {
            "video_name": video_name,
            "track_id": int(tid),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "length": int(len(rows)),
            "duration_sec": float(duration_sec),
            "dominant_class": dominant_class,
            "unique_classes": unique_classes,
            "class_switch_count": int(class_switch_count),
            "has_class_mutation": bool(has_class_mutation),
            "min_stable_frames": int(min_stable_frames),
            "class_switch_events": class_switch_events,
            "stable_class_names": stable_class_names,
            "stable_unique_classes": stable_unique_classes,
            "mean_conf": float(np.mean(confs)) if confs else 0.0,
            "min_conf": float(np.min(confs)) if confs else 0.0,
            "max_conf": float(np.max(confs)) if confs else 0.0,
            "mean_speed_px_per_sec": float(np.mean(speeds)) if speeds else 0.0,
            "max_speed_px_per_sec": float(np.max(speeds)) if speeds else 0.0,
            "total_distance_px": float(total_distance),
            "frames": frames,
            "boxes": boxes,
            "centers": centers,
            "class_ids": class_ids,
            "class_names": class_names,
            "confs": confs,
            "widths": widths,
            "heights": heights,
            "areas": areas,
            "velocities_px_per_sec": velocities,
            "speeds_px_per_sec": speeds,
            "frame_gaps": frame_gaps,
        }
        tracks_json.append(track_obj)
        summary_rows.append({
            "video_name": video_name,
            "track_id": int(tid),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "length": int(len(rows)),
            "duration_sec": float(duration_sec),
            "dominant_class": dominant_class,
            "unique_classes": "|".join(unique_classes),
            "class_switch_count": int(class_switch_count),
            "has_class_mutation": bool(has_class_mutation),
            "min_stable_frames": int(min_stable_frames),
            "stable_unique_classes": "|".join(stable_unique_classes),
            "mutation_events": json.dumps(class_switch_events, ensure_ascii=False),
            "mean_conf": float(np.mean(confs)) if confs else 0.0,
            "mean_speed_px_per_sec": float(np.mean(speeds)) if speeds else 0.0,
            "max_speed_px_per_sec": float(np.max(speeds)) if speeds else 0.0,
            "total_distance_px": float(total_distance),
        })
    tracks_json = sorted(tracks_json, key=lambda x: x["track_id"])
    summary_rows = sorted(summary_rows, key=lambda x: x["track_id"])
    return tracks_json, summary_rows
def main(
    video_path: str,
    weight_path: str,
    out_path: str = "out_tracked.mp4",
    conf: float = 0.25,
    iou: float = 0.5,
    device: int | str = 0,
    tracker: str = "bytetrack.yaml",
    tail_seconds: float = 2.0,
    vanish_after_seconds: float = 0.8,
    show_label: bool = True,
    save_fps: float | None = None,
    min_stable_frames: int = 3,
):

    model = YOLO(weight_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if save_fps is None:
        save_fps = fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(save_fps), (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {out_path}")
    maxlen = max(2, int(round(fps * tail_seconds)))
    vanish_after = max(1, int(round(fps * vanish_after_seconds)))
    trails = defaultdict(lambda: deque(maxlen=maxlen))
    last_seen = {}
    records = []
    pbar = tqdm(total=total if total > 0 else None, desc="Tracking", unit="frame")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        results = model.track(
            source=frame,
            conf=conf,
            iou=iou,
            device=device,
            persist=True,
            tracker=tracker,
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes
        if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
            if boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.full(len(xyxy), -1)
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.full(len(xyxy), -1.0)
                for j, bb in enumerate(xyxy):
                    x1, y1, x2, y2 = bb.astype(int)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    area = w * h
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    tid = int(ids[j])
                    cls_id = int(clss[j])
                    cls_name = model.names.get(cls_id, str(cls_id)) if cls_id >= 0 else "unknown_tool"
                    conf_value = float(confs[j])
                    trails[tid].append((cx, cy))
                    last_seen[tid] = frame_idx
                    # 逐帧记录：不再放在 show_label 内，否则 show_label=False 时不会保存轨迹
                    records.append({
                        "video_name": os.path.basename(video_path),
                        "frame": int(frame_idx),
                        "time_sec": float(frame_idx / fps) if fps > 0 else 0.0,
                        "track_id": tid,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "conf": conf_value,
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "w": int(w),
                        "h": int(h),
                        "area": int(area),
                        "cx": int(cx),
                        "cy": int(cy),
                    })
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)
                    if show_label:
                        txt = f"id:{tid} {cls_name} {conf_value:.2f}"
                        cv2.putText(
                            frame,
                            txt,
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
        # 消失目标只影响可视化拖尾；records 不会被删除，最终仍能得到完整历史轨迹
        to_del = [tid for tid, tlast in last_seen.items() if (frame_idx - tlast) > vanish_after]
        for tid in to_del:
            last_seen.pop(tid, None)
            trails.pop(tid, None)
        overlay = frame.copy()
        for tid, pts in trails.items():
            overlay = draw_fading_trail(overlay, list(pts), base_color=(0, 255, 255), max_thickness=6)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0.0)
        if total > 0:
            cv2.putText(
                frame,
                f"{frame_idx}/{total}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        writer.write(frame)
        pbar.update(1)
    pbar.close()
    cap.release()
    writer.release()
    base = out_path[:-4] if out_path.lower().endswith(".mp4") else out_path
    frame_csv_path = base + "_frame_records.csv"
    track_json_path = base + "_tracks.json"
    track_summary_csv_path = base + "_track_summary.csv"
    df = pd.DataFrame(records)
    df.to_csv(frame_csv_path, index=False)
    tracks_json, summary_rows = build_track_outputs(
        records=records,
        fps=fps,
        video_name=os.path.basename(video_path),
        model_names=model.names,
        min_stable_frames=min_stable_frames,
    )
    with open(track_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_name": os.path.basename(video_path),
            "video_path": video_path,
            "fps": float(fps),
            "width": int(W),
            "height": int(H),
            "tracker": tracker,
            "conf": conf,
            "iou": iou,
            "min_stable_frames": int(min_stable_frames),
            "num_tracks": len(tracks_json),
            "tracks": tracks_json,
        }, f, ensure_ascii=False, indent=2)
    pd.DataFrame(summary_rows).to_csv(track_summary_csv_path, index=False)
    print("Saved video:", out_path)
    print("Saved frame-level CSV:", frame_csv_path)
    print("Saved track-level JSON:", track_json_path)
    print("Saved track summary CSV:", track_summary_csv_path)
if __name__ == "__main__":
    main(
        video_path="/root/autodl-tmp/cholecTrack20/Testing/VID25/vid25.mp4",
        weight_path="../yolov26/runs/detect/runs_cholec/rtdetr_l_640-53/weights/best.pt",
        out_path="../out/vid25/vid25_tracked.mp4",
        conf=0.45,
        iou=0.5,
        device=0,
        tracker="bytetrack.yaml",  # 或 botsort.yaml
        tail_seconds=2.0,
        vanish_after_seconds=0.8,
        show_label=True,
        min_stable_frames=3,#不超过3帧的突变被视为抖动
    )