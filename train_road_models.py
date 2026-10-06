import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    print("=" * 65)
    print(" TRAINING ROAD ACCIDENT DETECTION CANDIDATES ON RTX 3050 GPU")
    print("=" * 65)

    if torch.cuda.is_available():
        device = 0
        dev_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Device: {dev_name} ({vram:.2f} GB VRAM) - CUDA Acceleration Active")
    else:
        device = "cpu"
        print("CUDA Acceleration: NOT AVAILABLE (Falling back to CPU)")

    road_yaml = Path(r"D:\accident\ML_project\data\road_cctv\data.yaml").as_posix()
    runs_dir = Path(r"D:\accident\ML_project\runs").as_posix()

    # Candidate 1: Fine-tune road_expanded_best.pt (YOLOv8n domain adaptation)
    print("\n" + "=" * 65)
    print(" CANDIDATE 1: Road v2 Nano (Fine-tune road_expanded_best.pt)")
    print("=" * 65)
    
    model_nano = YOLO("road_expanded_best.pt")
    res_nano = model_nano.train(
        data=road_yaml,
        epochs=15,
        imgsz=640,
        batch=16,
        device=device,
        workers=0,  # Windows safe
        patience=5,
        project=runs_dir,
        name="road_v2_nano",
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

    print("\nCandidate 1 Training Complete.")
    best_nano_weights = Path(runs_dir) / "road_v2_nano" / "weights" / "best.pt"
    print(f"Best Nano Weights: {best_nano_weights}")

    # Validate Candidate 1 on validation set
    val_nano = YOLO(best_nano_weights.as_posix())
    metrics_nano = val_nano.val(data=road_yaml, split="val", imgsz=640, device=device)

    print("\n" + "=" * 65)
    print(" CANDIDATE 2: Road v2 Small (yolov8s.pt)")
    print("=" * 65)

    model_small = YOLO("yolov8s.pt")
    res_small = model_small.train(
        data=road_yaml,
        epochs=15,
        imgsz=640,
        batch=8,  # batch 8 for 4GB VRAM safety with YOLOv8s
        device=device,
        workers=0,
        patience=5,
        project=runs_dir,
        name="road_v2_small",
        seed=42,
        deterministic=True,
        lr0=0.01,
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

    best_small_weights = Path(runs_dir) / "road_v2_small" / "weights" / "best.pt"
    print(f"Best Small Weights: {best_small_weights}")

    # Validate Candidate 2 on validation set
    val_small = YOLO(best_small_weights.as_posix())
    metrics_small = val_small.val(data=road_yaml, split="val", imgsz=640, device=device)

    print("\n" + "=" * 65)
    print(" VALIDATION COMPARISON (ROAD CANDIDATE MODELS)")
    print("=" * 65)
    print(f"{'Metric':<20} | {'Nano (v2)':<15} | {'Small (v2)':<15}")
    print("-" * 55)
    print(f"{'Precision':<20} | {metrics_nano.box.mp * 100:>13.2f}% | {metrics_small.box.mp * 100:>13.2f}%")
    print(f"{'Recall':<20} | {metrics_nano.box.mr * 100:>13.2f}% | {metrics_small.box.mr * 100:>13.2f}%")
    print(f"{'mAP@50':<20} | {metrics_nano.box.map50 * 100:>13.2f}% | {metrics_small.box.map50 * 100:>13.2f}%")
    print(f"{'mAP@50-95':<20} | {metrics_nano.box.map * 100:>13.2f}% | {metrics_small.box.map * 100:>13.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()
