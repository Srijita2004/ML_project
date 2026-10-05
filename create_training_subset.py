import os
import shutil
import random
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
SOURCE_DATASET = Path(r"D:\Fall_Expanded")
OUTPUT_SUBSET = Path(r"D:\Fall_Subset5k")

TARGET_OLD_FALL = 1500
TARGET_CAUCA_FALL = 1000
TARGET_CAUCA_NOFALL = 2500
TOTAL_TARGET = TARGET_OLD_FALL + TARGET_CAUCA_FALL + TARGET_CAUCA_NOFALL  # 5,000

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42

def find_images(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

def categorize_training_data():
    train_img_dir = SOURCE_DATASET / "train" / "images"
    train_lbl_dir = SOURCE_DATASET / "train" / "labels"

    all_images = find_images(train_img_dir)
    print(f"Total training images in source: {len(all_images)}")

    old_fall = []
    old_nofall = []
    cauca_fall = []
    cauca_nofall = []

    for img in all_images:
        lbl_file = train_lbl_dir / f"{img.stem}.txt"
        has_box = False
        if lbl_file.exists():
            text = lbl_file.read_text().strip()
            if text:
                has_box = True

        name = img.name
        if name.startswith("OLD_"):
            if has_box:
                old_fall.append(img)
            else:
                old_nofall.append(img)
        elif name.startswith("CAUCA_"):
            if has_box:
                cauca_fall.append(img)
            else:
                cauca_nofall.append(img)
        else:
            if has_box:
                old_fall.append(img)
            else:
                cauca_nofall.append(img)

    print("Source Categories:")
    print(f"  OLD Fall images       : {len(old_fall)}")
    print(f"  OLD No-fall images    : {len(old_nofall)}")
    print(f"  CAUCA Fall images     : {len(cauca_fall)}")
    print(f"  CAUCA No-fall images  : {len(cauca_nofall)}")

    return old_fall, cauca_fall, cauca_nofall

def create_subset():
    print("=" * 60)
    print("CREATING BALANCED 5,000-IMAGE TRAINING SUBSET")
    print("=" * 60)

    if not SOURCE_DATASET.exists():
        print(f"ERROR: {SOURCE_DATASET} does not exist!")
        return

    old_fall, cauca_fall, cauca_nofall = categorize_training_data()

    random.seed(RANDOM_SEED)

    sampled_old_fall = random.sample(old_fall, min(TARGET_OLD_FALL, len(old_fall)))
    sampled_cauca_fall = random.sample(cauca_fall, min(TARGET_CAUCA_FALL, len(cauca_fall)))
    sampled_cauca_nofall = random.sample(cauca_nofall, min(TARGET_CAUCA_NOFALL, len(cauca_nofall)))

    selected = sampled_old_fall + sampled_cauca_fall + sampled_cauca_nofall
    random.shuffle(selected)

    print(f"\nSampled Selection:")
    print(f"  OLD Fall selected     : {len(sampled_old_fall)}")
    print(f"  CAUCA Fall selected   : {len(sampled_cauca_fall)}")
    print(f"  CAUCA No-fall selected: {len(sampled_cauca_nofall)}")
    print(f"  TOTAL Selected        : {len(selected)}")

    dst_train_img = OUTPUT_SUBSET / "train" / "images"
    dst_train_lbl = OUTPUT_SUBSET / "train" / "labels"
    dst_train_img.mkdir(parents=True, exist_ok=True)
    dst_train_lbl.mkdir(parents=True, exist_ok=True)

    src_lbl_dir = SOURCE_DATASET / "train" / "labels"

    print("\nCopying selected images and labels to D:\\Fall_Subset5k...")
    copied_count = 0
    missing_labels = 0
    fall_instances = 0

    for img_path in selected:
        shutil.copy2(img_path, dst_train_img / img_path.name)

        src_lbl = src_lbl_dir / f"{img_path.stem}.txt"
        dst_lbl = dst_train_lbl / f"{img_path.stem}.txt"

        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
            text = src_lbl.read_text().strip()
            if text:
                for line in text.splitlines():
                    if line.strip().startswith("0 "):
                        fall_instances += 1
        else:
            dst_lbl.write_text("")
            missing_labels += 1

        copied_count += 1
        if copied_count % 1000 == 0:
            print(f"  Progress: {copied_count}/{len(selected)} images copied")

    print(f"\nFinished copying: {copied_count} images.")
    print(f"Total fall bounding boxes: {fall_instances}")
    print(f"Missing labels: {missing_labels}")

    yaml_content = f"""path: {OUTPUT_SUBSET.as_posix()}
train: train/images
val: {SOURCE_DATASET.as_posix()}/valid/images
test: {SOURCE_DATASET.as_posix()}/test/images

nc: 1
names:
  0: Fall-Detected
"""
    yaml_path = OUTPUT_SUBSET / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nCreated YAML configuration at: {yaml_path}")
    print(yaml_content)

    print("=" * 60)
    print("SUBSET CREATION COMPLETE")
    print("Validation and Test splits remain 100% untouched in D:\\Fall_Expanded.")
    print("=" * 60)

if __name__ == "__main__":
    create_subset()
