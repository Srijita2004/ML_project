import time
import json
from pathlib import Path
from app_fall import app

def run_tests():
    print("=" * 65)
    print(" RUNNING INTEGRATION TESTS FOR app_fall.py")
    print("=" * 65)

    client = app.test_client()

    # ----------------------------------------------------
    # TEST 1: Health Check Endpoint
    # ----------------------------------------------------
    print("\n[TEST 1] GET / (Health Check)")
    res = client.get("/")
    print(f"Status Code : {res.status_code}")
    print(f"Response    : {res.get_json()}")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert res.get_json().get("status") == "ok"
    print("PASSED")

    # ----------------------------------------------------
    # TEST 2: Empty Payload Handling
    # ----------------------------------------------------
    print("\n[TEST 2] POST /predict with empty payload")
    res = client.post("/predict", data=b"", content_type="application/octet-stream")
    print(f"Status Code : {res.status_code}")
    print(f"Response    : {res.get_json()}")
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    assert res.get_json().get("error") == "empty_body"
    print("PASSED")

    # ----------------------------------------------------
    # TEST 3: Positive Fall Emergency Detection
    # ----------------------------------------------------
    fall_sample = Path(r"D:\Fall_Expanded\test\images\CAUCA_cas1000079_png.rf.f58cc9429af6c1beff0b49cfc60b74b9.jpg")
    print(f"\n[TEST 3] POST /predict with FALL image: {fall_sample.name}")
    assert fall_sample.exists(), f"Sample image not found: {fall_sample}"

    img_bytes = fall_sample.read_bytes()

    start_t = time.time()
    res = client.post("/predict", data=img_bytes, content_type="application/octet-stream")
    elapsed_ms = (time.time() - start_t) * 1000

    data = res.get_json()
    print(f"Status Code : {res.status_code}")
    print(f"Latency     : {elapsed_ms:.2f} ms")
    print(f"Response    : {json.dumps(data, indent=2)}")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert data.get("emergency") is True, f"Expected emergency: True, got {data.get('emergency')}"
    assert data.get("type") == "fall", f"Expected type: fall, got {data.get('type')}"
    assert data.get("confidence") > 0.5, f"Expected confidence > 0.5, got {data.get('confidence')}"
    # Backward compatibility checks
    assert data.get("accident") is True
    print("PASSED (Fall detected with high confidence)")

    # ----------------------------------------------------
    # TEST 4: Negative Non-Fall Normal Activity
    # ----------------------------------------------------
    nofall_sample = Path(r"D:\Fall_Expanded\test\images\CAUCA_ars1000001_png.rf.5695d8e8198ebe5948d64ed46e5a0219.jpg")
    print(f"\n[TEST 4] POST /predict with NORMAL image: {nofall_sample.name}")
    assert nofall_sample.exists(), f"Sample image not found: {nofall_sample}"

    img_bytes = nofall_sample.read_bytes()

    start_t = time.time()
    res = client.post("/predict", data=img_bytes, content_type="application/octet-stream")
    elapsed_ms = (time.time() - start_t) * 1000

    data = res.get_json()
    print(f"Status Code : {res.status_code}")
    print(f"Latency     : {elapsed_ms:.2f} ms")
    print(f"Response    : {json.dumps(data, indent=2)}")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert data.get("emergency") is False, f"Expected emergency: False, got {data.get('emergency')}"
    assert data.get("type") == "none", f"Expected type: none, got {data.get('type')}"
    assert data.get("confidence") == 0.0, f"Expected confidence: 0.0, got {data.get('confidence')}"
    # Backward compatibility checks
    assert data.get("accident") is False
    print("PASSED (No false alarm)")

    # ----------------------------------------------------
    # TEST 5: Latency Benchmark (10 iterations)
    # ----------------------------------------------------
    print("\n[TEST 5] Inference Latency Benchmark (10 consecutive requests)")
    latencies = []
    for _ in range(10):
        t0 = time.time()
        client.post("/predict", data=img_bytes, content_type="application/octet-stream")
        latencies.append((time.time() - t0) * 1000)

    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    print(f"Min Latency : {min_lat:.2f} ms")
    print(f"Max Latency : {max_lat:.2f} ms")
    print(f"Avg Latency : {avg_lat:.2f} ms (approx {1000/avg_lat:.1f} FPS)")
    print("PASSED")

    print("\n" + "=" * 65)
    print(" ALL INTEGRATION TESTS PASSED SUCCESSFULLY! ")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
