import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

def calibrate_and_benchmark():
    print("=" * 70)
    print(" PHASE 10, 12, 13, 17, 18: CALIBRATION, BENCHMARK & ERROR ANALYSIS")
    print("=" * 70)

    road_val_yaml = r"D:\accident\ML_project\data\road_cctv\data.yaml"
    fall_val_yaml = r"D:\accident\ML_project\data\fall_dataset\data.yaml"

    old_road_path = "road_expanded_best.pt"
    new_road_path = r"D:\accident\ML_project\runs\road_v2_nano\weights\best.pt"

    old_fall_path = "fall_expanded_best.pt"
    new_fall_path = r"D:\accident\ML_project\runs\fall_v2_nano\weights\best.pt"

    old_road = YOLO(old_road_path)
    new_road = YOLO(new_road_path)

    old_fall = YOLO(old_fall_path)
    new_fall = YOLO(new_fall_path)

    # -------------------------------------------------------------
    # 1. CONFIDENCE CALIBRATION ON VALIDATION DATA (Road & Fall)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 1. CONFIDENCE CALIBRATION ON VALIDATION SETS")
    print("=" * 70)

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    print("\n[ROAD MODEL] Threshold Sweep on Validation Split:")
    print(f"{'Threshold':<10} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'FPR':<10}")
    print("-" * 65)

    best_road_thresh = 0.35
    best_road_f1 = 0.0

    for th in thresholds:
        # Validate with custom conf
        m = new_road.val(data=road_val_yaml, split="val", conf=th, imgsz=640, verbose=False)
        p = float(m.box.mp)
        r = float(m.box.mr)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        fpr = 1.0 - p  # approximate FP ratio
        print(f"{th:<10.2f} | {p*100:>10.2f}% | {r*100:>10.2f}% | {f1*100:>10.2f}% | {fpr*100:>8.2f}%")
        if f1 > best_road_f1:
            best_road_f1 = f1
            best_road_thresh = th

    print(f"\nOptimal Selected Road Confidence Threshold: {best_road_thresh:.2f} (F1: {best_road_f1*100:.2f}%)")

    print("\n[FALL MODEL] Threshold Sweep on Validation Split:")
    print(f"{'Threshold':<10} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'FPR':<10}")
    print("-" * 65)

    best_fall_thresh = 0.45
    best_fall_f1 = 0.0

    for th in thresholds:
        m = new_fall.val(data=fall_val_yaml, split="val", conf=th, imgsz=640, verbose=False)
        p = float(m.box.mp)
        r = float(m.box.mr)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        fpr = 1.0 - p
        print(f"{th:<10.2f} | {p*100:>10.2f}% | {r*100:>10.2f}% | {f1*100:>10.2f}% | {fpr*100:>8.2f}%")
        if f1 > best_fall_f1:
            best_fall_f1 = f1
            best_fall_thresh = th

    print(f"\nOptimal Selected Fall Confidence Threshold: {best_fall_thresh:.2f} (F1: {best_fall_f1*100:.2f}%)")

    # -------------------------------------------------------------
    # 2. INDEPENDENT FINAL TEST BENCHMARK (Untouched Held-Out Test)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 2. INDEPENDENT HELD-OUT TEST EVALUATION (EXACTLY ONCE)")
    print("=" * 70)

    # Road Test Benchmark: Old vs New
    print("\nRunning Road Test Set Evaluation (325 CCTV Images)...")
    res_road_old = old_road.val(data=road_val_yaml, split="test", conf=best_road_thresh, imgsz=640, verbose=False)
    res_road_new = new_road.val(data=road_val_yaml, split="test", conf=best_road_thresh, imgsz=640, verbose=False)

    # Fall Test Benchmark: Old vs New
    print("\nRunning Fall Test Set Evaluation (104 Images, including 30 Hard Negatives)...")
    res_fall_old = old_fall.val(data=fall_val_yaml, split="test", conf=best_fall_thresh, imgsz=640, verbose=False)
    res_fall_new = new_fall.val(data=fall_val_yaml, split="test", conf=best_fall_thresh, imgsz=640, verbose=False)

    # Measure CPU Latency (Cloud Production Sim)
    print("\nMeasuring CPU Inference Latency across models...")
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Force CPU for latency benchmark
    old_road_cpu = YOLO(old_road_path)
    new_road_cpu = YOLO(new_road_path)
    old_fall_cpu = YOLO(old_fall_path)
    new_fall_cpu = YOLO(new_fall_path)

    def measure_cpu_lat(m):
        lats = []
        for _ in range(15):
            t0 = time.perf_counter()
            _ = m(dummy_img, device="cpu", verbose=False)
            lats.append((time.perf_counter() - t0) * 1000)
        return float(np.mean(lats[3:]))  # warm-up 3 runs

    lat_road_old = measure_cpu_lat(old_road_cpu)
    lat_road_new = measure_cpu_lat(new_road_cpu)
    lat_fall_old = measure_cpu_lat(old_fall_cpu)
    lat_fall_new = measure_cpu_lat(new_fall_cpu)

    # Compile Final Benchmark Summary Table
    report = {
        "road": {
            "test_images": 325,
            "best_threshold": best_road_thresh,
            "old_model": {
                "precision": float(res_road_old.box.mp),
                "recall": float(res_road_old.box.mr),
                "map50": float(res_road_old.box.map50),
                "map50_95": float(res_road_old.box.map),
                "cpu_latency_ms": lat_road_old
            },
            "new_model": {
                "precision": float(res_road_new.box.mp),
                "recall": float(res_road_new.box.mr),
                "map50": float(res_road_new.box.map50),
                "map50_95": float(res_road_new.box.map),
                "cpu_latency_ms": lat_road_new
            }
        },
        "fall": {
            "test_images": 104,
            "best_threshold": best_fall_thresh,
            "old_model": {
                "precision": float(res_fall_old.box.mp),
                "recall": float(res_fall_old.box.mr),
                "map50": float(res_fall_old.box.map50),
                "map50_95": float(res_fall_old.box.map),
                "cpu_latency_ms": lat_fall_old
            },
            "new_model": {
                "precision": float(res_fall_new.box.mp),
                "recall": float(res_fall_new.box.mr),
                "map50": float(res_fall_new.box.map50),
                "map50_95": float(res_fall_new.box.map),
                "cpu_latency_ms": lat_fall_new
            }
        }
    }

    with open("ml_rebuild_benchmark_results.json", "w") as f_out:
        json.dump(report, f_out, indent=2)

    print("\n" + "=" * 70)
    print(" FINAL HEAD-TO-HEAD BENCHMARK: OLD VS NEW (HELD-OUT TEST SETS)")
    print("=" * 70)
    print(f"{'Task / Model':<22} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10} | {'mAP@50-95':<10} | {'CPU Latency':<12}")
    print("-" * 80)
    print(f"{'Road OLD (Baseline)':<22} | {report['road']['old_model']['precision']*100:>8.2f}% | {report['road']['old_model']['recall']*100:>8.2f}% | {report['road']['old_model']['map50']*100:>8.2f}% | {report['road']['old_model']['map50_95']*100:>8.2f}% | {lat_road_old:>8.2f} ms")
    print(f"{'Road NEW (v2 Nano)':<22} | {report['road']['new_model']['precision']*100:>8.2f}% | {report['road']['new_model']['recall']*100:>8.2f}% | {report['road']['new_model']['map50']*100:>8.2f}% | {report['road']['new_model']['map50_95']*100:>8.2f}% | {lat_road_new:>8.2f} ms")
    print("-" * 80)
    print(f"{'Fall OLD (Baseline)':<22} | {report['fall']['old_model']['precision']*100:>8.2f}% | {report['fall']['old_model']['recall']*100:>8.2f}% | {report['fall']['old_model']['map50']*100:>8.2f}% | {report['fall']['old_model']['map50_95']*100:>8.2f}% | {lat_fall_old:>8.2f} ms")
    print(f"{'Fall NEW (v2 Nano)':<22} | {report['fall']['new_model']['precision']*100:>8.2f}% | {report['fall']['new_model']['recall']*100:>8.2f}% | {report['fall']['new_model']['map50']*100:>8.2f}% | {report['fall']['new_model']['map50_95']*100:>8.2f}% | {lat_fall_new:>8.2f} ms")
    print("=" * 70)

if __name__ == "__main__":
    calibrate_and_benchmark()
