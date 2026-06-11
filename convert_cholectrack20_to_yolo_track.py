import os
import random
import re
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw
INSTRUMENTS = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
def tlwh_norm_to_yolo_cxcywh(tlwh_norm, img_w: int, img_h: int):
    x, y, bw, bh = map(float, tlwh_norm)
    x1 = clamp01(x)
    y1 = clamp01(y)
    x2 = clamp01(x + bw)
    y2 = clamp01(y + bh)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.0, 0.0, 0.0, 0.0
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy, bw, bh
def yolo_cxcywh_to_xyxy_px(cx, cy, bw, bh, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x1 = int(round((cx - bw / 2.0) * img_w))
    y1 = int(round((cy - bh / 2.0) * img_h))
    x2 = int(round((cx + bw / 2.0) * img_w))
    y2 = int(round((cy + bh / 2.0) * img_h))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    if x2 <= x1: x2 = min(img_w - 1, x1 + 1)
    if y2 <= y1: y2 = min(img_h - 1, y1 + 1)
    return x1, y1, x2, y2
# ------------------------------
# frame file matching
# ------------------------------
def extract_frame_id_from_name(name: str) -> Optional[int]:
    """
    Try to extract a frame index from filename.
    Works with patterns like:
      43976.jpg, frame_00043976.jpg, 00043976.png, VID103_43976.jpg ...
    Uses the last numeric group.
    """
    nums = re.findall(r"\d+", Path(name).stem)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None

def build_frame_map(frames_dir: Path) -> Dict[int, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    mapping: Dict[int, Path] = {}
    for p in sorted(frames_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        fid = extract_frame_id_from_name(p.name)
        if fid is None:
            continue
        # If duplicates exist, keep the first one encountered
        mapping.setdefault(fid, p)
    return mapping

# ------------------------------
# annotation parsing
# ------------------------------
def safe_instrument_name(idx) -> str:
    if idx is None:
        return "unknown"
    try:
        i = int(idx)
        if 0 <= i < len(INSTRUMENTS):
            return INSTRUMENTS[i]
        return f"instrument_{i}"
    except Exception:
        return "unknown"

def safe_instrument_class_id(idx) -> Optional[int]:
    if idx is None:
        return None
    try:
        i = int(idx)
        if 0 <= i < len(INSTRUMENTS):
            return i
        return None
    except Exception:
        return None

def get_obj_track_id(obj: dict) -> Optional[int]:
    # try common id fields
    for k in ["tool_id", "track_id", "instance_id", "id"]:
        if k in obj and obj[k] is not None:
            try:
                return int(obj[k])
            except Exception:
                pass
    return None

def objs_to_yolo_lines(
    objs: List[dict],
    img_w: int,
    img_h: int,
    include_id: bool = False,
) -> List[str]:
    """
    Convert objects in ONE frame to YOLO label lines.
    Uses tool_bbox only.
    """
    lines: List[str] = []
    for obj in objs:
        tlwh = obj.get("tool_bbox", None)
        if tlwh is None:
            continue

        cls = safe_instrument_class_id(obj.get("instrument", None))
        if cls is None:
            # skip unknown instrument types for YOLO training
            continue

        cx, cy, bw, bh = tlwh_norm_to_yolo_cxcywh(tlwh, img_w, img_h)
        if bw <= 0 or bh <= 0:
            continue

        if include_id:
            tid = get_obj_track_id(obj)
            if tid is None:
                # if missing, fallback to -1
                tid = -1
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {tid}")
        else:
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines

# ------------------------------
# io helpers
# ------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_or_link(src: Path, dst: Path, use_symlink: bool):
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if use_symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)

def write_text(path: Path, text: str):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()

def draw_vis(
    img_path: Path,
    yolo_lines: List[str],
    out_path: Path,
    class_names: List[str],
):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    dr = ImageDraw.Draw(img)

    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx = float(parts[1]); cy = float(parts[2]); bw = float(parts[3]); bh = float(parts[4])
        x1, y1, x2, y2 = yolo_cxcywh_to_xyxy_px(cx, cy, bw, bh, w, h)
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        dr.text((x1 + 2, y1 + 2), name, fill=(255, 255, 0))

    ensure_dir(out_path.parent)
    img.save(out_path)

def get_frame_objs(ann_dict, fid):
    fid_int = int(fid)
    candidates = [
        str(fid),
        str(fid_int),
        f"{fid_int:07d}",
        f"{fid_int:06d}",
        f"{fid_int:05d}",
        f"{fid_int:04d}",
    ]
    for k in candidates:
        if k in ann_dict:
            return ann_dict[k]
    return []

# ------------------------------
# main conversion
# ------------------------------
def convert_one_video(
    vid_dir: Path,
    split: str,
    out_root: Path,
    frames_dirname: str,
    json_name: Optional[str],
    include_id: bool,
    use_symlink: bool,
    save_vis: bool,
    vis_max: int,
) -> Dict[str, int]:
    """
    Convert one VIDxxx folder:
    - images -> out_root/images/{split}/VIDxxx/xxx.jpg
    - labels -> out_root/labels/{split}/VIDxxx/xxx.txt
    - optional vis -> out_root/vis/{split}/VIDxxx/xxx.jpg
    """
    frames_dir = vid_dir / frames_dirname
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

    # choose json
    if json_name is not None:
        jpath = vid_dir / json_name
        if not jpath.exists():
            raise FileNotFoundError(f"JSON not found: {jpath}")
    else:
        # pick first json in dir
        jsons = sorted([p for p in vid_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
        if not jsons:
            raise FileNotFoundError(f"No json found in {vid_dir} (use --json_name)")
        jpath = jsons[0]

    with open(jpath, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ann_dict = meta.get("annotations", {})

    frame_map = build_frame_map(frames_dir)
    if not frame_map:
        raise RuntimeError(f"No frame images found in {frames_dir}")

    # Determine all frame ids (save all images)
    all_fids = sorted(frame_map.keys())

    vid_name = vid_dir.name
    img_out_dir = out_root / "images" / split / vid_name
    lab_out_dir = out_root / "labels" / split / vid_name
    vis_out_dir = out_root / "vis" / split / vid_name

    stats = {
        "frames_total": 0,
        "frames_with_ann": 0,
        "boxes_total": 0,
        "boxes_written": 0,
        "images_copied": 0,
        "empty_labels": 0,
    }

    vis_saved = 0

    for fid in all_fids:
        img_src = frame_map[fid]
        objs = get_frame_objs(ann_dict, fid)

        stats["frames_total"] += 1
        if objs:
            stats["frames_with_ann"] += 1
        stats["boxes_total"] += len(objs)

        with Image.open(img_src) as im:
            w, h = im.size

        yolo_lines = objs_to_yolo_lines(objs, w, h, include_id=include_id)
        stats["boxes_written"] += len(yolo_lines)

        if len(yolo_lines) == 0:
            stats["empty_labels"] += 1
            continue

        img_dst = img_out_dir / img_src.name
        lab_dst = lab_out_dir / f"{img_src.stem}.txt"

        copy_or_link(img_src, img_dst, use_symlink=use_symlink)
        stats["images_copied"] += 1

        write_text(lab_dst, "\n".join(yolo_lines) + "\n")

    return stats

def find_vid_folders(root: Path) -> List[Path]:
    # accept root being VIDxxx itself
    if root.is_dir() and re.match(r"^VID\d+$", root.name, flags=re.IGNORECASE):
        return [root]

    vids = []
    for p in root.rglob("VID*"):
        if p.is_dir() and re.match(r"^VID\d+$", p.name, flags=re.IGNORECASE):
            vids.append(p)
    vids = sorted(set(vids))
    return vids

def read_yolo_label_classes(label_path: Path) -> List[int]:
    """
    Read class ids from a YOLO label txt.
    Empty label file returns [].
    """
    if not label_path.exists():
        return []

    classes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                classes.append(int(parts[0]))
            except Exception:
                continue
    return classes


def image_path_to_label_path(img_path: Path, out_root: Path) -> Path:
    """
    Convert:
      out_root/images/train/VIDxxx/000001.jpg
    to:
      out_root/labels/train/VIDxxx/000001.txt
    """
    rel = img_path.relative_to(out_root / "images" / "train")
    return out_root / "labels" / "train" / rel.with_suffix(".txt")


def build_oversampled_train_txt(
    out_root: Path,
    class_names: List[str],
    max_repeat: int = 6,
    min_repeat: int = 1,
    empty_repeat: int = 1,
    sqrt_balance: bool = True,
):
    """
    Build train_oversample.txt for long-tail class imbalance.

    Strategy:
    - Count class instances in labels/train
    - For each image, look at classes inside it
    - Image repeat factor = max repeat factor of classes appearing in that image
    - Minority-class images appear more often
    """

    train_img_dir = out_root / "images" / "train"
    train_label_dir = out_root / "labels" / "train"

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([
        p for p in train_img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ])

    if not image_paths:
        raise RuntimeError(f"No training images found under: {train_img_dir}")

    # 1) Count instances per class
    class_counts = {i: 0 for i in range(len(class_names))}
    image_to_classes = {}

    for img_path in image_paths:
        label_path = image_path_to_label_path(img_path, out_root)
        classes = read_yolo_label_classes(label_path)
        image_to_classes[img_path] = classes

        for c in classes:
            if c in class_counts:
                class_counts[c] += 1

    nonzero_counts = [v for v in class_counts.values() if v > 0]
    if not nonzero_counts:
        raise RuntimeError("No labeled boxes found in training labels.")

    max_count = max(nonzero_counts)

    # 2) Compute class repeat factor
    class_repeat = {}
    for c, count in class_counts.items():
        if count <= 0:
            class_repeat[c] = min_repeat
            continue

        ratio = max_count / count

        if sqrt_balance:
            repeat = int(round(ratio ** 0.5))
        else:
            repeat = int(round(ratio))

        repeat = max(min_repeat, min(max_repeat, repeat))
        class_repeat[c] = repeat

    print("\n[OVERSAMPLE] Class counts:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name:10s} count={class_counts[i]:6d}, repeat={class_repeat[i]}")

    # 3) Build oversampled image list
    oversampled_lines = []
    image_repeat_stats = {}

    for img_path in image_paths:
        classes = image_to_classes[img_path]

        if len(classes) == 0:
            repeat = empty_repeat
        else:
            unique_classes = set(classes)
            repeat = max(class_repeat.get(c, 1) for c in unique_classes)

        image_repeat_stats[img_path] = repeat

        # Ultralytics supports absolute image paths in txt
        for _ in range(repeat):
            oversampled_lines.append(img_path.as_posix())

    txt_path = out_root / "train_oversample.txt"
    random.seed(42)
    random.shuffle(oversampled_lines)

    write_text(txt_path, "\n".join(oversampled_lines) + "\n")

    print(f"\n[OVERSAMPLE] Original train images: {len(image_paths)}")
    print(f"[OVERSAMPLE] Oversampled train entries: {len(oversampled_lines)}")
    print(f"[OVERSAMPLE] Saved to: {txt_path}")

    # Optional: print image repeat distribution
    repeat_hist = {}
    for r in image_repeat_stats.values():
        repeat_hist[r] = repeat_hist.get(r, 0) + 1

    print("[OVERSAMPLE] Image repeat histogram:")
    for r in sorted(repeat_hist):
        print(f"  repeat {r}: {repeat_hist[r]} images")

    return txt_path

def write_data_yaml(out_root: Path, use_oversample: bool = False):
    yaml_path = out_root / "data.yaml"

    train_path = "train_oversample.txt" if use_oversample else "images/train"

    content = (
        f"path: {out_root.as_posix()}\n"
        f"train: {train_path}\n"
        f"val: images/val\n"
        f"names:\n"
    )

    for i, name in enumerate(INSTRUMENTS):
        content += f"  {i}: {name}\n"

    write_text(yaml_path, content)



def main():
    ap = argparse.ArgumentParser("Convert CholecTrack20 to YOLO labels (instrument boxes) for tracking/detection")
    ap.add_argument("--root", type=str, default='/root/autodl-tmp/cholecTrack20',
                    help="CholecTrack20 dataset root, or a VIDxxx folder, or Training/Testing parent.")
    ap.add_argument("--out_root", type=str, default='/root/autodl-tmp/cholecTrack20_yolo', help="Output YOLO dataset root")
    ap.add_argument("--training_subdir", type=str, default="Training", help="Training folder name under root")
    ap.add_argument("--testing_subdir", type=str, default="Validation", help="Testing folder name under root")
    ap.add_argument("--frames_dirname", type=str, default="Frames", help="Frames directory name under VIDxxx")
    ap.add_argument("--json_name", type=str, default=None,
                    help="Json filename inside VIDxxx (e.g., vid103.json). If None, use first .json found.")
    ap.add_argument("--include_id", action="store_true",
                    help="Append track_id/tool_id as 6th column for MOT-style labels (if available).")
    ap.add_argument("--symlink", action="store_true", help="Use symlink instead of copying images")
    ap.add_argument("--save_vis", action="store_true", help="Save a few visualization images with drawn boxes")
    ap.add_argument("--vis_max", type=int, default=50, help="Max number of vis images per split/video")
    ap.add_argument("--oversample", default=True,
                    help="Generate train_oversample.txt for class-balanced training.")

    ap.add_argument("--max_repeat", type=int, default=4,
                    help="Maximum repeat factor for minority-class images.")

    ap.add_argument("--empty_repeat", type=int, default=1,
                    help="Repeat factor for empty-label images.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    ensure_dir(out_root)

    # If user points to the dataset root that contains Training/Testing:
    train_root = root / args.training_subdir if (root / args.training_subdir).exists() else root
    val_root = root / args.testing_subdir if (root / args.testing_subdir).exists() else None

    # Convert Training -> train split
    vids_train = find_vid_folders(train_root)
    if not vids_train:
        raise RuntimeError(f"No VIDxxx folders found under: {train_root}")

    print(f"[INFO] Found {len(vids_train)} train videos under {train_root}")

    total_stats_train = {"frames_total": 0, "frames_with_ann": 0, "boxes_total": 0,
                         "boxes_written": 0, "images_copied": 0, "empty_labels": 0}
    for vid in vids_train:
        s = convert_one_video(
            vid_dir=vid,
            split="train",
            out_root=out_root,
            frames_dirname=args.frames_dirname,
            json_name=args.json_name,
            include_id=args.include_id,
            use_symlink=args.symlink,
            save_vis=args.save_vis,
            vis_max=args.vis_max,
        )
        for k in total_stats_train:
            total_stats_train[k] += s[k]

    print("[TRAIN] stats:", total_stats_train)

    # Convert Testing -> val split (if exists)
    if val_root is not None and val_root.exists():
        vids_val = find_vid_folders(val_root)
        print(f"[INFO] Found {len(vids_val)} val videos under {val_root}")

        total_stats_val = {"frames_total": 0, "frames_with_ann": 0, "boxes_total": 0,
                           "boxes_written": 0, "images_copied": 0, "empty_labels": 0}
        for vid in vids_val:
            s = convert_one_video(
                vid_dir=vid,
                split="val",
                out_root=out_root,
                frames_dirname=args.frames_dirname,
                json_name=args.json_name,
                include_id=args.include_id,
                use_symlink=args.symlink,
                save_vis=args.save_vis,
                vis_max=args.vis_max,
            )
            for k in total_stats_val:
                total_stats_val[k] += s[k]
        print("[VAL] stats:", total_stats_val)
    else:
        print("[INFO] No Testing folder found; only train split is generated.")

    # data.yaml
    # Oversampling
    if args.oversample:
        build_oversampled_train_txt(
            out_root=out_root,
            class_names=INSTRUMENTS,
            max_repeat=args.max_repeat,
            empty_repeat=args.empty_repeat,
            sqrt_balance=True,
        )

    # data.yaml
    write_data_yaml(out_root, use_oversample=args.oversample)

    print(f"[DONE] YOLO dataset saved to: {out_root}")
    print(f"[DONE] data.yaml saved: {out_root / 'data.yaml'}")

    if args.oversample:
        print(f"[DONE] oversampled train txt saved: {out_root / 'train_oversample.txt'}")

if __name__ == "__main__":
    main()
    #剔除验证集空标签
    from pathlib import Path
    root = Path("/root/autodl-tmp/cholecTrack20_yolo")
    val_img_dir = root / "images" / "val"
    val_label_dir = root / "labels" / "val"
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    lines = []
    for img in sorted(val_img_dir.rglob("*")):
        if img.suffix.lower() not in exts:
            continue
        rel = img.relative_to(val_img_dir)
        lb = val_label_dir / rel.with_suffix(".txt")
        if lb.exists() and lb.read_text().strip():
            lines.append(img.as_posix())
    out = root / "val_nonempty.txt"
    out.write_text("\n".join(lines) + "\n")
    print("saved:", out)
    print("non-empty val images:", len(lines))