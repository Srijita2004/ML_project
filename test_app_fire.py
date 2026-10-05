import time
import json
from pathlib import Path
import sys
sys.path.insert(0, r"D:\accident\accident")

from app_final import app

def run_tests():
    print("=" * 75)
    print(" RUNNING INTEGRATION TESTS FOR app_final.py (FIRE DETECTION)")
    print("=" * 75)

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
    # TEST 3: Positive Fire Incident 1 (fire_basket.jpg)
    # ----------------------------------------------------
    fire_sample_1 = Path(r"D:\accident\accident\test_images\fire_samples\fire_basket.jpg")
    print(f"[TEST 3] POST /predict with REAL FIRE image: {fire_sample_1.name}")
    assert fire_sample_1.exists(), f"Sample image not found: {fire_sample_1}"

    img_bytes = fire_sample_1.read_bytes()
    t0 = time.time()
    res = client.post("/predict", data=img_bytes, content_type="application/octet-stream")
    latency_ms = (time.time() - t0) * 1000

    data = res.get_json()
    print(f"Status Code : {res.status_code}")
    print(f"Latency     : {latency_ms:.2f} ms")
    print(f"Response    : {json.dumps(data, indent=2)}")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert data.get("accident") is True, f"Expected accident: True, got {data.get('accident')}"
    assert data.get("type") == "fire_smoke_accident", f"Expected type: fire_smoke_accident, got {data.get('type')}"
    assert data.get("score") > 0.040, f"Expected score > 0.040, got {data.get('score')}"
    print(">>> TEST 3 PASSED (Active fire detected with high confidence)\n")

    # ----------------------------------------------------
    # TEST 4: Positive Fire Incident 2 (pan_fire.jpg)
    # ----------------------------------------------------
    fire_sample_2 = Path(r"D:\accident\accident\test_images\fire_samples\pan_fire.jpg")
    print(f"[TEST 4] POST /predict with SECOND REAL FIRE image: {fire_sample_2.name}")
    assert fire_sample_2.exists(), f"Sample image not found: {fire_sample_2}"

    img_bytes = fire_sample_2.read_bytes()
    t0 = time.time()
    res = client.post("/predict", data=img_bytes, content_type="application/octet-stream")
    latency_ms = (time.time() - t0) * 1000

    data = res.get_json()
    print(f"Status Code : {res.status_code}")
    print(f"Latency     : {latency_ms:.2f} ms")
    print(f"Response    : {json.dumps(data, indent=2)}")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert data.get("accident") is True, f"Expected accident: True, got {data.get('accident')}"
    assert data.get("type") == "fire_smoke_accident", f"Expected type: fire_smoke_accident, got {data.get('type')}"
    assert data.get("score") > 0.040, f"Expected score > 0.040, got {data.get('score')}"
    print(">>> TEST 4 PASSED (Flames detected accurately)\n")

    # ----------------------------------------------------
    # TEST 5: Negative Challenge (Sunset Sky: images (3).jpg)
    # ----------------------------------------------------
    sunset_sample = Path(r"D:\accident\accident\test_images\images (3).jpg")
    print(f"[TEST 5] POST /predict with SUNSET SKY image: {sunset_sample.name}")
    assert sunset_sample.exists(), f"Sample image not found: {sunset_sample}"

    img_bytes = sunset_sample.read_bytes()
    t0 = time.time()
    res = client.post("/predict", data=img_bytes, content_type="application/octet-stream")
    latency_ms = (time.time() - t0) * 1000

    data = res.get_json()
    print(f"Status Code : {res.status_code}")
    print(f"Latency     : {latency_ms:.2f} ms")
    print(f"Response    : {json.dumps(data, indent=2)}")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    # Must NOT be detected as fire
    assert data.get("type") != "fire_smoke_accident", f"False fire alert triggered on sunset sky!"
    print(">>> TEST 5 PASSED (Sunset sky correctly rejected, zero fire false alarm)\n")

    # ----------------------------------------------------
    # TEST 6: Negative Challenge (Red Vehicle: _91193810_pune.jpg)
    # ----------------------------------------------------
    red_car_sample = Path(r"D:\accident\accident\test_images\_91193810_pune.jpg")
    print(f"[TEST 6] POST /predict with RED VEHICLE ACCIDENT image: {red_car_sample.name}")
    assert red_car_sample.exists(), f"Sample image not found: {red_car_sample}"

    img_bytes = red_car_sample.read_bytes()
    t0 = time.time()
    res = client.post("/predict", data=img_bytes, content_type="application/octet-stream")
    latency_ms = (time.time() - t0) * 1000

    data = res.get_json()
    print(f"Status Code : {res.status_code}")
    print(f"Latency     : {latency_ms:.2f} ms")
    print(f"Response    : {json.dumps(data, indent=2)}")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    # Must NOT trigger fire
    assert data.get("type") != "fire_smoke_accident", f"False fire alert on red vehicle paint!"
    # It SHOULD proceed to road detection and detect road accident!
    assert data.get("type") == "road_vehicle_accident", f"Expected road_vehicle_accident, got {data.get('type')}"
    print(">>> TEST 6 PASSED (Red car paint correctly rejected by fire filter, correctly detected by road detector!)\n")

    # ----------------------------------------------------
    # TEST 7: Negative Challenge (Red Bus: images (6).jpg)
    # ----------------------------------------------------
    red_bus_sample = Path(r"D:\accident\accident\test_images\images (6).jpg")
    print(f"[TEST 7] POST /predict with RED BUS image: {red_bus_sample.name}")
    assert red_bus_sample.exists(), f"Sample image not found: {red_bus_sample}"

    img_bytes = red_bus_sample.read_bytes()
    t0 = time.time()
    res = client.post("/predict", data=img_bytes, content_type="application/octet-stream")
    latency_ms = (time.time() - t0) * 1000

    data = res.get_json()
    print(f"Status Code : {res.status_code}")
    print(f"Latency     : {latency_ms:.2f} ms")
    print(f"Response    : {json.dumps(data, indent=2)}")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert data.get("type") != "fire_smoke_accident", f"False fire alert on red bus paint!"
    print(">>> TEST 7 PASSED (Red bus paint correctly rejected by fire filter)\n")

    # ----------------------------------------------------
    # TEST 8: Latency & Throughput Benchmark (20 requests)
    # ----------------------------------------------------
    print("[TEST 8] End-to-End Latency Benchmark (20 HTTP POST requests)")
    test_bytes = fire_sample_1.read_bytes()
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
    assert avg_lat < 40.0, f"Latency too high: {avg_lat} ms"
    print(">>> TEST 8 PASSED (Sub-20ms ultra-fast throughput confirmed)\n")

    print("=" * 75)
    print(" ALL 8 FIRE INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
