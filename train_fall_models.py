import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    print("=" * 65)
    print(" TRAINING FALL DETECTION MODEL WITH HARD-NEGATIVE REJECTION")
    print("=" * 65)

    if torch.cuda.is_available():
        device = 0
        dev_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Device: {dev_name} ({vram:.2f} GB VRAM) - CUDA Acceleration Active")
    else:
        device = "cpu"
        print("CUDA Acceleration: NOT AVAILABLE (Falling back to CPU)")

    fall_yaml = Path(r"D:\accident\ML_project\data\fall_dataset\data.yaml").as_posix()
    runs_dir = Path(r"D:\accident\ML_project\runs").as_posix()

    model = YOLO("fall_expanded_best.pt")
    results = model.train(
        data=fall_yaml,
        epochs=20,
        imgsz=640,
        batch=16,
        device=device,
        workers=0,
        patience=7,
        project=runs_dir,
        name="fall_v2_nano",
        seed=42,
        deterministic=True,
        lr0=0.005,
        lrf=0.01,
        amp=True,
        mosaic=0.5,
        mixup=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        save=True,
        plots=True,
        verbose=True
    )

    best_weights = Path(runs_dir) / "fall_v2_nano" / "weights" / "best.pt"
    print(f"\nBest Fall Weights: {best_weights}")

    # Validate on validation set
    val_model = YOLO(best_weights.as_posix())
    metrics = val_model.val(data=fall_yaml, split="val", imgsz=640, device=device)

    print("\n" + "=" * 65)
    print(" FALL MODEL VALIDATION METRICS")
    print("=" * 65)
    print(f"Precision : {metrics.box.mp * 100:.2f}%")
    print(f"Recall    : {metrics.box.mr * 100:.2f}%")
    print(f"mAP@50    : {metrics.box.map50 * 100:.2f}%")
    print(f"mAP@50-95 : {metrics.box.map * 100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()
