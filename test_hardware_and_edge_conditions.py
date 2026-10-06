import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
from calibration import PlattCalibrator

def apply_phone_pipeline(img, mode="standard"):
    h, w = img.shape[:2]
    out = img.copy()

    if mode == "portrait":
        # Rotate or crop to 9:16 portrait
        if w > h:
            new_w = int(h * 9 / 16)
            start_x = (w - new_w) // 2
            out = out[:, max(0, start_x):min(w, start_x + new_w)]
    elif mode == "low_light":
        # Simulates nighttime / dim indoor lighting
        table = np.array([((i / 255.0) ** 1.8) * 255 for i in np.arange(0, 256)]).astype("uint8")
        out = cv2.LUT(out, table)
        out = cv2.convertScaleAbs(out, alpha=0.7, beta=-20)
    elif mode == "bright_sunlight":
        # High dynamic range / sun glare
        table = np.array([((i / 255.0) ** 0.7) * 255 for i in np.arange(0, 256)]).astype("uint8")
        out = cv2.LUT(out, table)
        out = cv2.convertScaleAbs(out, alpha=1.2, beta=20)
    elif mode == "motion_blur":
        # Simulates hand tremor / moving phone
        size = 9
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size
        out = cv2.filter2D(out, -1, kernel)
    elif mode == "high_compression":
        # Lower quality phone network streaming
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 40])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Standard browser canvas compression (quality 0.85)
    _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def apply_esp32_pipeline(img):
    """
    Simulates ESP32-CAM OV2640 sensor:
    - 320x240 QVGA
    - JPEG Quality 15
    - Sensor noise (sigma=10)
    - Low dynamic range
    """
    # 1. Downscale to 320x240
    qvga = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)

    # 2. Add sensor noise
    noise = np.random.normal(0, 10, qvga.shape).astype(np.float32)
    noisy = np.clip(qvga.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 3. Low dynamic range contrast crush
    crushed = cv2.convertScaleAbs(noisy, alpha=0.9, beta=5)

    # 4. Aggressive ESP32 JPEG compression (quality 15)
    _, buf = cv2.imencode(".jpg", crushed, [cv2.IMWRITE_JPEG_QUALITY, 15])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def run_pipeline_test():
    print("=" * 75)
    print(" REAL PHONE-CAMERA & ESP32-CAM HARDWARE DEGRADATION EVALUATION")
    print("=" * 75)

    base_dir = r"D:\accident\ML_project"
    fire_model = YOLO(os.path.join(base_dir, "models", "production", "fire_v2_yolo11s.pt"))
    road_model = YOLO(os.path.join(base_dir, "models", "production", "road_v3_small_best.pt"))
    fall_model = YOLO(os.path.join(base_dir, "models", "production", "fall_v2_hardneg_best.pt"))

    def detect_emergency(frame):
        # 1. Fire
        res_fire = fire_model(frame, conf=0.30, verbose=False)
        best_fire = max([float(b.conf[0].item()) for b in res_fire[0].boxes] + [0.0])

        # 2. Road
        res_road = road_model(frame, conf=0.30, verbose=False)
        best_road = 0.0
        if res_road[0].boxes:
            for b in res_road[0].boxes:
                c_id = int(b.cls[0].item())
                if road_model.names.get(c_id, "") in ["human_incident", "vehicle_incident"]:
                    cf = float(b.conf[0].item())
                    if cf > best_road: best_road = cf

        # 3. Fall
        res_fall = fall_model(frame, conf=0.30, verbose=False)
        best_fall = max([float(b.conf[0].item()) for b in res_fall[0].boxes] + [0.0])

        is_emg = (best_fire >= 0.45) or (best_road >= 0.40) or (best_fall >= 0.45)
        scores = {"fire": best_fire, "road": best_road, "fall": best_fall}
        return is_emg, scores

    # Test images suite
    test_suite = [
        ("test_images/skynews-car-crash-goodmayes_7250187.jpg", True, "road"),
        ("test_images/Pedestrian-accident-3.jpg", True, "road"),
        ("test_images/_91193810_pune.jpg", True, "road"),
        ("test_images/images (1).jpg", True, "road"),
        ("test_images/fall_sample.jpg", True, "fall"),
        ("test_images/fire_samples/fire_basket.jpg", True, "fire"),
        ("test_images/fire_samples/pan_fire.jpg", True, "fire"),
        ("test_images/images (3).jpg", False, "normal"),
        ("test_images/images (4).jpg", False, "normal"),
        ("test_images/images (5).jpg", False, "normal"),
        ("test_images/images.jpg", False, "normal"),
    ]

    modes = ["standard", "portrait", "low_light", "bright_sunlight", "motion_blur", "high_compression"]
    phone_results = {}

    for mode in modes:
        correct = 0
        total = 0
        for path, gt_emg, hazard in test_suite:
            if not os.path.exists(path):
                continue
            orig = cv2.imread(path)
            processed = apply_phone_pipeline(orig, mode=mode)
            pred_emg, scores = detect_emergency(processed)
            if pred_emg == gt_emg:
                correct += 1
            total += 1
        acc = (correct / total) * 100 if total > 0 else 0
        phone_results[mode] = {"accuracy": acc, "correct": correct, "total": total}

    print("\n[PHONE-CAMERA PIPELINE RESULTS]")
    print(f"{'Condition':<20} | {'Samples':<10} | {'Accuracy':<12}")
    print("-" * 50)
    for m, res in phone_results.items():
        print(f"{m:<20} | {res['total']:<10} | {res['accuracy']:>10.1f}%")

    # ESP32-CAM Test
    esp_correct = 0
    esp_total = 0
    esp_hazard_hits = {}

    for path, gt_emg, hazard in test_suite:
        if not os.path.exists(path):
            continue
        orig = cv2.imread(path)
        esp_frame = apply_esp32_pipeline(orig)
        pred_emg, scores = detect_emergency(esp_frame)
        if pred_emg == gt_emg:
            esp_correct += 1
        if gt_emg:
            esp_hazard_hits[hazard] = esp_hazard_hits.get(hazard, []) + [pred_emg]
        esp_total += 1

    esp_acc = (esp_correct / esp_total) * 100 if esp_total > 0 else 0
    print("\n[ESP32-CAM DEGRADED PIPELINE (320x240, Q15, Noise)]")
    print(f"Overall Accuracy : {esp_acc:.1f}% ({esp_correct}/{esp_total})")
    for hz, hits in esp_hazard_hits.items():
        rec = (sum(hits) / len(hits)) * 100
        print(f"  {hz.capitalize()} Recall under ESP32 degradation: {rec:.1f}% ({sum(hits)}/{len(hits)})")

    out_file = "hardware_degradation_benchmark.json"
    with open(out_file, "w") as f:
        json.dump({
            "phone_pipeline": phone_results,
            "esp32_pipeline": {
                "accuracy": esp_acc,
                "hazard_recalls": {hz: (sum(h)/len(h))*100 for hz, h in esp_hazard_hits.items()}
            }
        }, f, indent=2)
    print(f"\nSaved hardware degradation results to {out_file}")

if __name__ == "__main__":
    run_pipeline_test()
