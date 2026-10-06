import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

def test_production_realism():
    print("=" * 65)
    print(" PHASE 15 & 16: PHONE CAMERA & ESP32-CAM REALISM TESTS")
    print("=" * 65)

    road_model_path = r"D:\accident\ML_project\runs\road_v2_nano\weights\best.pt"
    fall_model_path = r"D:\accident\ML_project\runs\fall_v2_nano\weights\best.pt"

    road_model = YOLO(road_model_path)
    fall_model = YOLO(fall_model_path)

    test_dir = Path(r"D:\accident\ML_project\test_images")
    sample_images = list(test_dir.glob("*.jpg"))
    print(f"Testing on {len(sample_images)} real-world project test images...")

    results_table = []

    for img_p in sample_images:
        img_orig = cv2.imread(str(img_p))
        if img_orig is None:
            continue

        h, w = img_orig.shape[:2]

        # 1. Standard Phone Camera Input
        t0 = time.perf_counter()
        res_road = road_model(img_orig, conf=0.35, verbose=False)
        lat_phone = (time.perf_counter() - t0) * 1000

        road_incident = False
        road_conf = 0.0
        for r in res_road:
            if r.boxes and len(r.boxes) > 0:
                for b in r.boxes:
                    cname = road_model.names.get(int(b.cls[0].item()), "")
                    conf = float(b.conf[0].item())
                    if cname in ["vehicle_incident", "human_incident"]:
                        road_incident = True
                        if conf > road_conf:
                            road_conf = conf

        # 2. Simulated ESP32-CAM Input (QVGA 320x240, JPEG quality 25, high compression)
        esp32_img = cv2.resize(img_orig, (320, 240))
        _, enc = cv2.imencode(".jpg", esp32_img, [cv2.IMWRITE_JPEG_QUALITY, 25])
        esp32_degraded = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        t0 = time.perf_counter()
        res_esp = road_model(esp32_degraded, conf=0.35, verbose=False)
        lat_esp = (time.perf_counter() - t0) * 1000

        esp_incident = False
        esp_conf = 0.0
        for r in res_esp:
            if r.boxes and len(r.boxes) > 0:
                for b in r.boxes:
                    cname = road_model.names.get(int(b.cls[0].item()), "")
                    conf = float(b.conf[0].item())
                    if cname in ["vehicle_incident", "human_incident"]:
                        esp_incident = True
                        if conf > esp_conf:
                            esp_conf = conf

        results_table.append({
            "image": img_p.name,
            "phone_detected": road_incident,
            "phone_conf": road_conf,
            "phone_latency_ms": lat_phone,
            "esp_detected": esp_incident,
            "esp_conf": esp_conf,
            "esp_latency_ms": lat_esp
        })

    print(f"\n{'Image Name':<35} | {'Phone AI (Conf)':<16} | {'ESP32-CAM (Conf)':<16}")
    print("-" * 72)
    for r in results_table:
        p_str = f"YES ({r['phone_conf']:.2f})" if r['phone_detected'] else "NO (0.00)"
        e_str = f"YES ({r['esp_conf']:.2f})" if r['esp_detected'] else "NO (0.00)"
        print(f"{r['image'][:34]:<35} | {p_str:<16} | {e_str:<16}")

    avg_p_lat = np.mean([r['phone_latency_ms'] for r in results_table])
    avg_e_lat = np.mean([r['esp_latency_ms'] for r in results_table])
    print("\nLatency Benchmarks:")
    print(f"  Phone High-Res Stream Latency : {avg_p_lat:.2f} ms")
    print(f"  ESP32-CAM Low-Res Frame Latency : {avg_e_lat:.2f} ms")
    print("=" * 65)

if __name__ == "__main__":
    test_production_realism()
