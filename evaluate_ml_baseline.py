import os
import glob
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO

# Ground truth definition for the benchmark dataset
BENCHMARK_GROUND_TRUTH = {
    # Road Crashes
    "test_images/skynews-car-crash-goodmayes_7250187.jpg": {"accident": True, "type": "ROAD_ACCIDENT"},
    "test_images/_91193810_pune.jpg": {"accident": True, "type": "ROAD_ACCIDENT"},
    "test_images/Pedestrian-accident-3.jpg": {"accident": True, "type": "ROAD_ACCIDENT"},
    
    # Human Falls
    "test_images/images (1).jpg": {"accident": True, "type": "FALL_ACCIDENT"},
    "test_images/images (2).jpg": {"accident": True, "type": "FALL_ACCIDENT"},
    
    # Fire Incidents
    "test_images/fire_samples/fire_basket.jpg": {"accident": True, "type": "FIRE_ACCIDENT"},
    "test_images/fire_samples/pan_fire.jpg": {"accident": True, "type": "FIRE_ACCIDENT"},
    
    # Normal / Non-Accident Scenes (Traffic, Parked Cars, Pedestrians, Sky)
    "test_images/images (3).jpg": {"accident": False, "type": "NORMAL"},
    "test_images/images (4).jpg": {"accident": False, "type": "NORMAL"},
    "test_images/images (5).jpg": {"accident": False, "type": "NORMAL"},
    "test_images/images (6).jpg": {"accident": False, "type": "NORMAL"},
    "test_images/images.jpg": {"accident": False, "type": "NORMAL"}
}

def is_fire_like(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if h == 0 or w == 0:
        return 0.0, False, False
    if max(h, w) > 640:
        scale = 640.0 / max(h, w)
        frame_eval = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        h_eval, w_eval = frame_eval.shape[:2]
    else:
        frame_eval = frame_bgr
        h_eval, w_eval = h, w
    tot_pixels = h_eval * w_eval
    if tot_pixels == 0:
        return 0.0, False, False

    hsv = cv2.cvtColor(frame_eval, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = cv2.split(hsv)
    b, g, r = cv2.split(frame_eval)
    ycbcr = cv2.cvtColor(frame_eval, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    flame_mask = (
        ((h_c <= 28) | (h_c >= 165)) &
        (s_c >= 95) & (v_c >= 155) &
        (r >= 180) & (r > g) & (g > b) &
        ((r.astype(int) - b.astype(int)) >= 70) &
        ((cr.astype(int) - cb.astype(int)) >= 38)
    ).astype(np.uint8) * 255

    core_mask = (
        (v_c >= 215) & (r >= 215) & (g >= 120) &
        ((r.astype(int) - b.astype(int)) >= 85)
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(flame_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(50, int(tot_pixels * 0.0003))

    valid_fire_pixels = 0
    max_blob_area = 0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y_box, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(1, ch)
        if aspect > 3.0:
            continue
        mask_roi = np.zeros((h_eval, w_eval), dtype=np.uint8)
        cv2.drawContours(mask_roi, [c], -1, 255, -1)
        core_in_blob = np.count_nonzero(core_mask & (mask_roi > 0))
        if (core_in_blob / area) < 0.05 and area > 350:
            continue
        valid_fire_pixels += area
        if area > max_blob_area:
            max_blob_area = area

    fire_ratio = valid_fire_pixels / tot_pixels
    max_blob_ratio = max_blob_area / tot_pixels
    strong_hit = (fire_ratio >= 0.040) or (max_blob_ratio >= 0.035)
    medium_hit = (fire_ratio >= 0.026) or (max_blob_ratio >= 0.018)
    return fire_ratio, strong_hit, medium_hit

def run_baseline_prediction(img, road_model, fall_model):
    """Evaluates frame using old/baseline thresholds."""
    h, w = img.shape[:2]
    frame_area = h * w if h > 0 and w > 0 else 1

    # 1. Fire
    fire_ratio, strong_hit, medium_hit = is_fire_like(img)
    if strong_hit:
        return {"accident": True, "type": "FIRE_ACCIDENT", "score": float(fire_ratio)}

    # 2. Road (Old: conf=0.35, immediate trigger)
    road_results = road_model(img, conf=0.35, verbose=False)
    for r in road_results:
        if r.boxes is not None and len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].item())
                conf = float(r.boxes.conf[i].item())
                class_name = road_model.names.get(cls_id, "")
                if class_name in ["human_incident", "vehicle_incident"] and conf >= 0.35:
                    return {"accident": True, "type": "ROAD_ACCIDENT", "score": conf}

    # 3. Fall (Old: conf=0.80, min_area=0.08)
    fall_results = fall_model(img, conf=0.80, verbose=False)
    for r in fall_results:
        if r.boxes is not None and len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                conf = float(r.boxes.conf[i].item())
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                if conf >= 0.80 and (box_area / frame_area) >= 0.08:
                    return {"accident": True, "type": "FALL_ACCIDENT", "score": conf}

    return {"accident": False, "type": "NORMAL", "score": 0.0}

def run_refined_prediction(img, road_model, fall_model):
    """Evaluates frame using refined, calibrated multi-modal synthesis."""
    h, w = img.shape[:2]
    frame_area = h * w if h > 0 and w > 0 else 1

    # 1. Fire Filter
    fire_ratio, strong_hit, medium_hit = is_fire_like(img)
    if strong_hit:
        return {"accident": True, "type": "FIRE_ACCIDENT", "score": float(fire_ratio)}

    # 2. Multi-model parallel detection
    road_results = road_model(img, conf=0.30, verbose=False)
    fall_results = fall_model(img, conf=0.30, verbose=False)

    best_road_conf = 0.0
    best_road_label = None
    normal_vehicle_max_conf = 0.0
    normal_human_max_conf = 0.0

    for r in road_results:
        if r.boxes is not None and len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].item())
                conf = float(r.boxes.conf[i].item())
                cname = road_model.names.get(cls_id, "")
                if cname in ["human_incident", "vehicle_incident"]:
                    if conf > best_road_conf:
                        best_road_conf = conf
                        best_road_label = cname
                elif cname == "vehicle_normal":
                    if conf > normal_vehicle_max_conf:
                        normal_vehicle_max_conf = conf
                elif cname == "human_normal":
                    if conf > normal_human_max_conf:
                        normal_human_max_conf = conf

    best_fall_conf = 0.0
    for r in fall_results:
        if r.boxes is not None and len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                conf = float(r.boxes.conf[i].item())
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                area_ratio = box_area / frame_area
                # Calibrated fall threshold: conf >= 0.50, min area >= 0.01 (1%)
                if conf >= 0.50 and area_ratio >= 0.01:
                    if conf > best_fall_conf:
                        best_fall_conf = conf

    # Decision synthesis:
    # A. Check Fall First if Fall is strong and higher than Road
    if best_fall_conf >= 0.50 and best_fall_conf >= best_road_conf:
        return {"accident": True, "type": "FALL_ACCIDENT", "score": best_fall_conf}

    # B. Check Road Accident:
    # Require best_road_conf >= 0.48. If there are strong normal vehicles (>0.85),
    # ensure incident is not a faint false trigger.
    if best_road_conf >= 0.48:
        # If best_road_label is human_incident and fall is also detected, classify based on stronger score
        if best_road_label == "human_incident" and best_fall_conf >= 0.50:
            if best_fall_conf > best_road_conf:
                return {"accident": True, "type": "FALL_ACCIDENT", "score": best_fall_conf}
        return {"accident": True, "type": "ROAD_ACCIDENT", "score": best_road_conf}

    # C. Fall fallback if road wasn't high enough
    if best_fall_conf >= 0.50:
        return {"accident": True, "type": "FALL_ACCIDENT", "score": best_fall_conf}

    return {"accident": False, "type": "NORMAL", "score": 0.0}

def compute_metrics(results, ground_truth):
    tp = fp = tn = fn = 0
    type_matches = 0
    total_positives = 0

    for path, gt in ground_truth.items():
        pred = results[path]
        gt_acc = gt["accident"]
        pred_acc = pred["accident"]

        if gt_acc and pred_acc:
            tp += 1
            total_positives += 1
            if gt["type"] == pred["type"]:
                type_matches += 1
        elif not gt_acc and pred_acc:
            fp += 1
        elif not gt_acc and not pred_acc:
            tn += 1
        elif gt_acc and not pred_acc:
            fn += 1
            total_positives += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    type_accuracy = type_matches / total_positives if total_positives > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Accuracy": round(accuracy, 4),
        "TypeClassificationAccuracy": round(type_accuracy, 4)
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    road_path = os.path.join(base_dir, "road_expanded_best.pt")
    fall_path = os.path.join(base_dir, "fall_expanded_best.pt")

    print(f"Loading Road Model: {road_path}")
    road_model = YOLO(road_path)
    print(f"Loading Fall Model: {fall_path}")
    fall_model = YOLO(fall_path)

    # Normalize paths
    gt_map = {}
    for rel_path, meta in BENCHMARK_GROUND_TRUTH.items():
        full_path = os.path.normpath(os.path.join(base_dir, rel_path))
        gt_map[full_path] = meta

    baseline_preds = {}
    refined_preds = {}

    print("\n" + "=" * 70)
    print("EVALUATING TEST DATASET (12 IMAGES)")
    print("=" * 70)

    for full_path, meta in gt_map.items():
        img = cv2.imread(full_path)
        assert img is not None, f"Could not load {full_path}"
        base_pred = run_baseline_prediction(img, road_model, fall_model)
        ref_pred = run_refined_prediction(img, road_model, fall_model)
        baseline_preds[full_path] = base_pred
        refined_preds[full_path] = ref_pred

        b_name = os.path.basename(full_path)
        print(f"\nImage: {b_name}")
        print(f"  Ground Truth : {meta}")
        print(f"  Baseline Pred: {base_pred}")
        print(f"  Refined Pred : {ref_pred}")

    base_metrics = compute_metrics(baseline_preds, gt_map)
    ref_metrics = compute_metrics(refined_preds, gt_map)

    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Baseline':<15} | {'Refined':<15}")
    print("-" * 60)
    for k in base_metrics:
        print(f"{k:<25} | {str(base_metrics[k]):<15} | {str(ref_metrics[k]):<15}")

    # Output JSON for reporting
    report = {
        "baseline": base_metrics,
        "refined": ref_metrics
    }
    with open(os.path.join(base_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nEvaluation report saved to evaluation_report.json")

if __name__ == "__main__":
    main()
