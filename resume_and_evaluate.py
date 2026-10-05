import os
import torch
from ultralytics import YOLO

def main():
    print("=" * 65)
    print(" RESUMING FALL DETECTION TRAINING & BENCHMARK")
    print("=" * 65)

    last_pt = r"D:\accident\accident\runs\fall_subset5k_v1\weights\last.pt"
    best_pt = r"D:\accident\accident\runs\fall_subset5k_v1\weights\best.pt"
    test_yaml = r"D:\Fall_Expanded\data.yaml"
    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)} (CUDA:0)")
    else:
        print("Device: CPU")

    if not os.path.exists(last_pt):
        print(f"ERROR: {last_pt} not found!")
        return

    print(f"\nResuming training from: {last_pt}")
    model = YOLO(last_pt)
    model.train(resume=True)

    print("\n" + "=" * 65)
    print("TRAINING FINISHED — STARTING TEST EVALUATION (5,036 IMAGES)")
    print("=" * 65)

    eval_weights = best_pt if os.path.exists(best_pt) else last_pt
    eval_model = YOLO(eval_weights)

    val_results = eval_model.val(
        data=test_yaml,
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
    print("FINAL BENCHMARK: BASELINE vs. EXPANDED SUBSET MODEL")
    print("=" * 65)
    print(f"{'Metric':<15} | {'Baseline (Original)':<20} | {'Improved Model':<20} | {'Gain':<10}")
    print("-" * 72)
    print(f"{'Precision':<15} | {'74.4%':<20} | {p * 100:>18.2f}% | {(p - 0.744) * 100:>+8.2f}%")
    print(f"{'Recall':<15} | {'71.3%':<20} | {r * 100:>18.2f}% | {(r - 0.713) * 100:>+8.2f}%")
    print(f"{'mAP@50':<15} | {'76.6%':<20} | {map50 * 100:>18.2f}% | {(map50 - 0.766) * 100:>+8.2f}%")
    print(f"{'mAP@50-95':<15} | {'44.4%':<20} | {map50_95 * 100:>18.2f}% | {(map50_95 - 0.444) * 100:>+8.2f}%")
    print("=" * 65)

    print("\nEvaluation complete. Best weights saved at:", eval_weights)

if __name__ == "__main__":
    main()
