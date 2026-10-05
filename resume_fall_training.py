import os
import torch
from ultralytics import YOLO

def main():
    print("=" * 65)
    print(" RESUME FALL DETECTION TRAINING / EVALUATION")
    print("=" * 65)

    last_weights = r"D:\accident\accident\runs\fall_subset5k_v1\weights\last.pt"
    best_weights = r"D:\accident\accident\runs\fall_subset5k_v1\weights\best.pt"
    test_yaml = r"D:\Fall_Expanded\data.yaml"
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Last Checkpoint : {last_weights}")
    print(f"Best Checkpoint : {best_weights}")

    choice = input("\nOptions:\n1. Resume remaining epochs (Epochs 21-25)\n2. Evaluate current best model on untouched test set\nEnter 1 or 2 (default 1): ").strip()

    if choice == "2":
        print("\nEvaluating best.pt on untouched test set (5,036 images)...")
        model = YOLO(best_weights)
        val_results = model.val(data=test_yaml, split="test", imgsz=640, device=device)
        print("Test evaluation finished.")
    else:
        print("\nResuming training from last.pt...")
        model = YOLO(last_weights)
        model.train(resume=True)

if __name__ == "__main__":
    main()
