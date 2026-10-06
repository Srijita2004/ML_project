import time
import json
from pathlib import Path
from ultralytics import YOLO

# Import app from app_final
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app_final
from app_final import app

def run_tests():
    print("=" * 75)
    print(" RUNNING INTEGRATION TESTS FOR app_final.py (ROAD ACCIDENT DETECTION)")
    print("=" * 75)

    # Verify app_final loaded upgraded road model natively
    print(f"[TEST SETUP] Verifying app_final.road_model_path: {app_final.road_model_path}")
    assert "road_expanded_best.pt" in str(app_final.road_model_path), f"Expected road_expanded_best.pt, got {app_final.road_model_path}"
    print("[TEST SETUP] Verified native deployment of road_expanded_best.pt in app_final.py.\n")

    client = app.test_client()

    # ----------------------------------------------------
    # TEST 1: Health Check Endpoint
    # ----------------------------------------------------
    print("[TEST 1] GET / (Health Check)")
    res = client.get("/")
    print(f"Status Code : {res.status_code}")
    print(f"Response    : {res.get_json()}")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert res.get_json().get("status") == "ok"
    print(">>> TEST 1 PASSED\n")

    # ----------------------------------------------------
    # TEST 2: Empty Payload Validation
    # ----------------------------------------------------
    print("[TEST 2] POST /predict with empty payload")
    res = client.post("/predict", data=b"", content_type="application/octet-stream")
    print(f"Status Code : {res.status_code}")
    print(f"Response    : {res.get_json()}")
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    assert res.get_json().get("error") == "empty_body"
    print(">>> TEST 2 PASSED\n")

    # ----------------------------------------------------
    # TEST 3: Positive Vehicle Accident Detection
    # ----------------------------------------------------
    car_crash_sample = Path(__file__).resolve().parent / "test_images" / "skynews-car-crash-goodmayes_7250187.jpg"
    print(f"[TEST 3] POST /predict with CAR CRASH image: {car_crash_sample.name}")
    assert car_crash_sample.exists(), f"Sample image not found: {car_crash_sample}"

    img_bytes = car_crash_sample.read_bytes()
    t0 = time.time()
    res = client.post("/predict", data=img_bytes, content_type="application/octet-stream")
    latency_ms = (time.time() - t0) * 1000

    data = res.get_json()
    print(f"Status Code : {res.status_code}")
    print(f"Latency     : {latency_ms:.2f} ms")
    print(f"Response    : {json.dumps(data, indent=2)}")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert data.get("accident") is True, f"Expected accident: True, got {data.get('accident')}"
    assert data.get("type") == "road_vehicle_accident", f"Expected type: road_vehicle_accident, got {data.get('type')}"
    assert data.get("score") > 0.50, f"Expected confidence > 0.50, got {data.get('score')}"
    assert "vehicle_incident" in data.get("labels", []), "Expected 'vehicle_incident' in labels"
    print(">>> TEST 3 PASSED (Vehicle collision detected with high confidence)\n")

    # ----------------------------------------------------
    # TEST 4: Positive Pedestrian Accident Detection
    # ----------------------------------------------------
    ped_sample = Path(__file__).resolve().parent / "test_images" / "Pedestrian-accident-3.jpg"
    print(f"[TEST 4] POST /predict with PEDESTRIAN ACCIDENT image: {ped_sample.name}")
    assert ped_sample.exists(), f"Sample image not found: {ped_sample}"

    img_bytes = ped_sample.read_bytes()
    t0 = time.time()
    res = client.post("/predict", data=img_bytes, content_type="application/octet-stream")
    latency_ms = (time.time() - t0) * 1000

    data = res.get_json()
    print(f"Status Code : {res.status_code}")
    print(f"Latency     : {latency_ms:.2f} ms")
    print(f"Response    : {json.dumps(data, indent=2)}")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert data.get("accident") is True, f"Expected accident: True, got {data.get('accident')}"
    assert data.get("type") == "road_vehicle_accident", f"Expected type: road_vehicle_accident, got {data.get('type')}"
    assert data.get("score") > 0.50, f"Expected confidence > 0.50, got {data.get('score')}"
    assert "human_incident" in data.get("labels", []), "Expected 'human_incident' in labels"
    print(">>> TEST 4 PASSED (Pedestrian hit/incident detected successfully)\n")

    # ----------------------------------------------------
    # TEST 5: Negative Normal Traffic Scene (No False Alarm)
    # ----------------------------------------------------
    normal_sample = Path(__file__).resolve().parent / "test_images" / "images (4).jpg"
    print(f"[TEST 5] POST /predict with NORMAL TRAFFIC image: {normal_sample.name}")
    assert normal_sample.exists(), f"Sample image not found: {normal_sample}"

    img_bytes = normal_sample.read_bytes()
    t0 = time.time()
    res = client.post("/predict", data=img_bytes, content_type="application/octet-stream")
    latency_ms = (time.time() - t0) * 1000

    data = res.get_json()
    print(f"Status Code : {res.status_code}")
    print(f"Latency     : {latency_ms:.2f} ms")
    print(f"Response    : {json.dumps(data, indent=2)}")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert data.get("accident") is False, f"Expected accident: False, got {data.get('accident')}"
    assert data.get("type") == "non_accident", f"Expected type: non_accident, got {data.get('type')}"
    print(">>> TEST 5 PASSED (Normal traffic correctly classified, zero false alarm)\n")

    # ----------------------------------------------------
    # TEST 6: Second Normal Traffic Scene
    # ----------------------------------------------------
    normal_sample_2 = Path(__file__).resolve().parent / "test_images" / "images.jpg"
    print(f"[TEST 6] POST /predict with SECOND NORMAL TRAFFIC image: {normal_sample_2.name}")
    assert normal_sample_2.exists(), f"Sample image not found: {normal_sample_2}"

    img_bytes = normal_sample_2.read_bytes()
    t0 = time.time()
    res = client.post("/predict", data=img_bytes, content_type="application/octet-stream")
    latency_ms = (time.time() - t0) * 1000

    data = res.get_json()
    print(f"Status Code : {res.status_code}")
    print(f"Latency     : {latency_ms:.2f} ms")
    print(f"Response    : {json.dumps(data, indent=2)}")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert data.get("accident") is False, f"Expected accident: False, got {data.get('accident')}"
    print(">>> TEST 6 PASSED (No false alarm on normal vehicles)\n")

    # ----------------------------------------------------
    # TEST 7: End-to-End Latency Benchmark (20 requests)
    # ----------------------------------------------------
    print("[TEST 7] Latency & Throughput Benchmark (20 HTTP POST requests)")
    test_bytes = car_crash_sample.read_bytes()
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        client.post("/predict", data=test_bytes, content_type="application/octet-stream")
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    fps = 1000.0 / avg_lat

    print(f"Min Latency : {min_lat:5.2f} ms")
    print(f"Avg Latency : {avg_lat:5.2f} ms")
    print(f"Max Latency : {max_lat:5.2f} ms")
    print(f"Throughput  : {fps:5.1f} FPS (Frames Per Second)")
    assert avg_lat < 100.0, f"Latency too high: {avg_lat} ms"
    print(">>> TEST 7 PASSED (Real-time speed requirements met)\n")

    print("=" * 75)
    print(" ALL 7 INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
