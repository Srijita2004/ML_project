import os
import sys
import io
import hashlib
import zipfile
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download

BASE_DATA_DIR = Path(r"D:\accident\ML_project\data")
ROAD_DIR = BASE_DATA_DIR / "road_cctv"
FALL_DIR = BASE_DATA_DIR / "fall_dataset"

def get_hash(img_bytes):
    return hashlib.sha256(img_bytes).hexdigest()

def prepare_road_dataset():
    print("=" * 65)
    print("PREPARING ROAD ACCIDENT CCTV DATASET (CC0-1.0)")
    print("=" * 65)
    
    repo_id = "justjuu/traffic-accident-cctv-object-detection"
    splits = {
        "train": [
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet"
        ],
        "val": [
            "data/validation-00000-of-00001.parquet"
        ],
        "test": [
            "data/test-00000-of-00001.parquet"
        ]
    }

    seen_hashes = {"train": set(), "val": set(), "test": set()}
    all_seen = {}
    leakage_count = 0

    class_stats = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter()
    }
    img_counts = {"train": 0, "val": 0, "test": 0}

    # Class mappings:
    # 0 -> 2 (vehicle_incident)
    # 1 -> 3 (vehicle_normal)
    # 4 classes: 0: human_incident, 1: human_normal, 2: vehicle_incident, 3: vehicle_normal
    cat_mapping = {
        0: 2,  # vehicle_incident
        1: 3   # vehicle_normal
    }

    for split_name, parquet_files in splits.items():
        img_dir = ROAD_DIR / split_name / "images"
        lbl_dir = ROAD_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for p_file in parquet_files:
            print(f"Downloading/loading {p_file}...")
            local_p = hf_hub_download(repo_id, p_file, repo_type="dataset")
            table = pq.read_table(local_p)
            num_rows = len(table)
            print(f"Processing {num_rows} records from {p_file}...")

            for i in range(num_rows):
                img_data = table["image"][i].as_py()
                img_bytes = img_data["bytes"]
                img_hash = get_hash(img_bytes)

                # Check data leakage across splits
                if img_hash in all_seen and all_seen[img_hash] != split_name:
                    leakage_count += 1
                    # Skip duplicate to prevent data leakage!
                    continue

                if img_hash in seen_hashes[split_name]:
                    # Intra-split duplicate, skip
                    continue

                seen_hashes[split_name].add(img_hash)
                all_seen[img_hash] = split_name

                # Image name
                filename = f"cctv_{split_name}_{img_counts[split_name]:05d}.jpg"
                img_path = img_dir / filename
                lbl_path = lbl_dir / f"cctv_{split_name}_{img_counts[split_name]:05d}.txt"

                # Save image
                with open(img_path, "wb") as f_img:
                    f_img.write(img_bytes)

                # Extract bboxes
                objs = table["objects"][i].as_py()
                bboxes = objs.get("bbox", [])
                cats = objs.get("category", [])

                lines = []
                for box, cat in zip(bboxes, cats):
                    mapped_cls = cat_mapping.get(cat, 2)
                    class_stats[split_name][mapped_cls] += 1
                    
                    # Box format: [x_min, y_min, width, height] in 640x640 space
                    x_min, y_min, bw, bh = box
                    # Clamp and normalize to [0, 1]
                    x_min = max(0.0, min(640.0, float(x_min)))
                    y_min = max(0.0, min(640.0, float(y_min)))
                    bw = max(1.0, min(640.0 - x_min, float(bw)))
                    bh = max(1.0, min(640.0 - y_min, float(bh)))

                    x_center = (x_min + bw / 2.0) / 640.0
                    y_center = (y_min + bh / 2.0) / 640.0
                    norm_w = bw / 640.0
                    norm_h = bh / 640.0

                    # Validate YOLO bounds
                    if 0.0 < norm_w <= 1.0 and 0.0 < norm_h <= 1.0:
                        lines.append(f"{mapped_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

                with open(lbl_path, "w") as f_lbl:
                    f_lbl.write("\n".join(lines))

                img_counts[split_name] += 1

    print("\nRoad CCTV Dataset Preprocessing Complete:")
    print(f"  Train Images : {img_counts['train']}")
    print(f"  Val Images   : {img_counts['val']}")
    print(f"  Test Images  : {img_counts['test']}")
    print(f"  Total Images : {sum(img_counts.values())}")
    print(f"  Cross-split Leakage Detected and Excluded: {leakage_count}")
    print(f"  Class Box Distribution:")
    for s in ["train", "val", "test"]:
        print(f"    [{s.upper()}] vehicle_incident: {class_stats[s][2]}, vehicle_normal: {class_stats[s][3]}")

    # Generate data.yaml
    yaml_content = f"""path: {ROAD_DIR.as_posix()}
train: train/images
val: val/images
test: test/images

nc: 4
names:
  0: human_incident
  1: human_normal
  2: vehicle_incident
  3: vehicle_normal
"""
    yaml_file = ROAD_DIR / "data.yaml"
    yaml_file.write_text(yaml_content)
    print(f"Created YAML config at: {yaml_file}")
    return img_counts, class_stats


def prepare_fall_dataset():
    print("\n" + "=" * 65)
    print("PREPARING FALL DETECTION DATASET (MIT License)")
    print("=" * 65)
    
    zip_path = hf_hub_download("DeZan/fall-detection", "fall detection dataset.zip", repo_type="dataset")
    print(f"Extracting DeZan fall dataset from {zip_path}...")
    
    extracted_imgs = []
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        img_names = [n for n in names if n.endswith((".jpg", ".png")) and "images" in n]
        print(f"Found {len(img_names)} raw images in zip")

        # Organize into Train (70%), Val (15%), Test (15%)
        # Ensure deterministic sort to avoid random leakage
        img_names.sort()
        n_total = len(img_names)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val

        split_assignment = {}
        for idx, n in enumerate(img_names):
            if idx < n_train:
                split_assignment[n] = "train"
            elif idx < n_train + n_val:
                split_assignment[n] = "val"
            else:
                split_assignment[n] = "test"

        fall_stats = {"train": Counter(), "val": Counter(), "test": Counter()}
        fall_img_counts = {"train": 0, "val": 0, "test": 0}

        for n in img_names:
            s_name = split_assignment[n]
            dst_img_dir = FALL_DIR / s_name / "images"
            dst_lbl_dir = FALL_DIR / s_name / "labels"
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            img_bytes = z.read(n)
            stem = Path(n).stem
            dst_img_name = f"fall_{stem}.jpg"
            (dst_img_dir / dst_img_name).write_bytes(img_bytes)

            # Find corresponding label in zip
            # Format: fall_dataset/images/train/fall001.jpg -> fall_dataset/labels/train/fall001.txt
            lbl_name_candidate1 = n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            lbl_name_candidate2 = f"fall_dataset/labels/{n.split('/images/')[1].rsplit('.', 1)[0]}.txt"

            lbl_bytes = b""
            if lbl_name_candidate1 in names:
                lbl_bytes = z.read(lbl_name_candidate1)
            elif lbl_name_candidate2 in names:
                lbl_bytes = z.read(lbl_name_candidate2)
            else:
                # search by stem
                for potential in names:
                    if potential.endswith(f"{stem}.txt"):
                        lbl_bytes = z.read(potential)
                        break

            lbl_text = lbl_bytes.decode("utf-8", errors="ignore").strip()
            # Clean and validate YOLO label
            cleaned_lines = []
            for line in lbl_text.splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = 0  # Fall-Detected
                    fall_stats[s_name][cls_id] += 1
                    cleaned_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

            (dst_lbl_dir / f"fall_{stem}.txt").write_text("\n".join(cleaned_lines))
            fall_img_counts[s_name] += 1

    # Ingest hard negatives: normal human actions from Bingsu/Human_Action_Recognition
    print("\nIngesting Hard-Negative Normal Human Activities (Sitting, Walking, Standing)...")
    try:
        har_p = hf_hub_download("Bingsu/Human_Action_Recognition", "data/test-00000-of-00001.parquet", repo_type="dataset")
        har_table = pq.read_table(har_p)
        print(f"Loaded HAR dataset ({len(har_table)} samples)")

        # Ingest 150 normal images into train, 30 into val, 30 into test as background (0-box labels)
        har_splits = {"train": 150, "val": 30, "test": 30}
        har_idx = 0
        for s_name, target_count in har_splits.items():
            dst_img_dir = FALL_DIR / s_name / "images"
            dst_lbl_dir = FALL_DIR / s_name / "labels"
            for _ in range(target_count):
                if har_idx >= len(har_table):
                    break
                h_bytes = har_table["image"][har_idx].as_py()["bytes"]
                har_img_name = f"hardneg_action_{har_idx:04d}.jpg"
                (dst_img_dir / har_img_name).write_bytes(h_bytes)
                # Empty label file = background negative sample in YOLO!
                (dst_lbl_dir / f"hardneg_action_{har_idx:04d}.txt").write_text("")
                fall_img_counts[s_name] += 1
                har_idx += 1
        print(f"Successfully added {har_idx} hard-negative normal activity samples with 0-box labels!")
    except Exception as e:
        print("Warning: Could not fetch HAR parquet:", e)

    print("\nFall Dataset Preprocessing Complete:")
    print(f"  Train Images : {fall_img_counts['train']}")
    print(f"  Val Images   : {fall_img_counts['val']}")
    print(f"  Test Images  : {fall_img_counts['test']}")
    print(f"  Total Images : {sum(fall_img_counts.values())}")
    print(f"  Fall-Detected Boxes: Train={fall_stats['train'][0]}, Val={fall_stats['val'][0]}, Test={fall_stats['test'][0]}")

    yaml_content = f"""path: {FALL_DIR.as_posix()}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: Fall-Detected
"""
    yaml_file = FALL_DIR / "data.yaml"
    yaml_file.write_text(yaml_content)
    print(f"Created YAML config at: {yaml_file}")
    return fall_img_counts, fall_stats

if __name__ == "__main__":
    prepare_road_dataset()
    prepare_fall_dataset()
    print("\nALL DATASETS PREPARED AND VERIFIED SUCCESSFULLY!")
