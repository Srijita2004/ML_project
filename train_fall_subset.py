import os
import torch
from ultralytics import YOLO

def main():
    print("=" * 65)
    print(" UNIVERSAL GOLDEN MINUTE - FALL DETECTION FINE-TUNING")
    print("=" * 65)

    # 1. Device check
    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA ACCELERATION: AVAILABLE")
        print(f"Device: {device_name} ({vram:.2f} GB VRAM)")
    else:
        device = "cpu"
        print("CUDA ACCELERATION: NOT AVAILABLE (Falling back to CPU)")

    # 2. Paths
    baseline_model_path = r"D:\accident\accident\fall_accident_model_best.pt"
    dataset_yaml = r"D:\Fall_Subset5k\data.yaml"
    untouched_test_yaml = r"D:\Fall_Expanded\data.yaml"
    project_dir = r"D:\accident\accident\runs"
    run_name = "fall_subset5k_v1"

    if not os.path.exists(baseline_model_path):
        print(f"ERROR: Baseline model not found at {baseline_model_path}")
        return

    if not os.path.exists(dataset_yaml):
        print(f"ERROR: Subset data configuration not found at {dataset_yaml}")
        return

    print(f"\nBaseline Model (Untouched) : {baseline_model_path}")
    print(f"Training Dataset (Subset)   : {dataset_yaml}")
    print(f"Output Run Directory        : {os.path.join(project_dir, run_name)}")

    # 3. Load baseline weights
    print("\nLoading baseline model weights...")
    model = YOLO(baseline_model_path)

    # 4. Start Fine-Tuning
    print("\n" + "=" * 65)
    print("STARTING TRAINING (25 Epochs, Batch 8, Image Size 640)")
    print("=" * 65)

    results = model.train(
        data=dataset_yaml,
        epochs=25,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        patience=7,
        project=project_dir,
        name=run_name,
        pretrained=True,
        plots=True,
        cache=False
    )

    print("\n" + "=" * 65)
    print("TRAINING FINISHED")
    print("=" * 65)

    best_weights = os.path.join(project_dir, run_name, "weights", "best.pt")
    if not os.path.exists(best_weights):
        print("Could not find best.pt, trying last.pt...")
        best_weights = os.path.join(project_dir, run_name, "weights", "last.pt")

    # 5. Evaluate on untouched test set
    print("\n" + "=" * 65)
    print("EVALUATING NEW MODEL ON UNTOUCHED TEST SET (5,036 IMAGES)")
    print("=" * 65)

    eval_model = YOLO(best_weights)
    val_results = eval_model.val(
        data=untouched_test_yaml,
        split="test",
        imgsz=640,
        batch=8 if device == 0 else 4,
        device=device
    )

    p = val_results.box.p[0] if len(val_results.box.p) > 0 else val_results.box.mp
    r = val_results.box.r[0] if len(val_results.box.r) > 0 else val_results.box.mr
    map50 = val_results.box.map50
    map50_95 = val_results.box.map

    print("\n" + "=" * 65)
    print("BENCHMARK COMPARISON: BASELINE vs. EXPANDED SUBSET MODEL")
    print("=" * 65)
    print(f"{'Metric':<15} | {'Baseline (Original)':<20} | {'Improved Model':<20} | {'Diff':<10}")
    print("-" * 72)
    print(f"{'Precision':<15} | {'74.4%':<20} | {p * 100:>18.2f}% | {(p - 0.744) * 100:>+8.2f}%")
    print(f"{'Recall':<15} | {'71.3%':<20} | {r * 100:>18.2f}% | {(r - 0.713) * 100:>+8.2f}%")
    print(f"{'mAP@50':<15} | {'76.6%':<20} | {map50 * 100:>18.2f}% | {(map50 - 0.766) * 100:>+8.2f}%")
    print(f"{'mAP@50-95':<15} | {'44.4%':<20} | {map50_95 * 100:>18.2f}% | {(map50_95 - 0.444) * 100:>+8.2f}%")
    print("=" * 65)

    print("\nBest model saved at:", best_weights)
    print("Original baseline model remains completely untouched.")

if __name__ == "__main__":
    main()
