import requests
import time
import os
from pathlib import Path

SPACE_URL = "https://srij1-esp32-accident-brain.hf.space"
PREDICT_URL = f"{SPACE_URL}/predict"

def check_space():
    print("=" * 80)
    print(f" TESTING LIVE HUGGING FACE SPACE: {SPACE_URL}")
    print("=" * 80)

    # 1. Health / Home check
    print("\n[STEP 1] Checking Space Root Endpoint (GET /)...")
    for attempt in range(12):
        try:
            r = requests.get(SPACE_URL, timeout=10)
            if r.status_code == 200:
                print(f"[OK] Space is ONLINE! Response: {r.json()}")
                break
            else:
                print(f"Attempt {attempt+1}: Status code {r.status_code}, waiting...")
        except Exception as e:
            print(f"Attempt {attempt+1}: Space rebuilding/starting ({e})...")
        time.sleep(5)
    else:
        print("Warning: Space still initializing after 60 seconds.")

    # Find smoke test sample
    smoke_sample = None
    fire_test_dir = r"data\fire_test\test\images"
    fire_test_lbl = r"data\fire_test\test\labels"
    if os.path.exists(fire_test_dir):
        for f in os.listdir(fire_test_dir):
            lbl = os.path.join(fire_test_lbl, f.replace(".jpg", ".txt").replace(".png", ".txt"))
            if os.path.exists(lbl):
                with open(lbl) as lf:
                    for line in lf:
                        if line.strip().startswith("1 "):
                            smoke_sample = os.path.join(fire_test_dir, f)
                            break
            if smoke_sample:
                break

    # Find hard-negative normal person sample
    hardneg_sample = None
    fall_test_dir = r"data\fall_dataset\test\images"
    if os.path.exists(fall_test_dir):
        for f in os.listdir(fall_test_dir):
            if f.startswith("hardneg"):
                hardneg_sample = os.path.join(fall_test_dir, f)
                break

    test_scenarios = [
        ("1. Obvious Fire (Fire Basket)", "test_images/fire_samples/fire_basket.jpg", "fire"),
        ("2. Smaller Pan Fire", "test_images/fire_samples/pan_fire.jpg", "fire"),
        ("3. Smoke Sample", smoke_sample, "fire"),
        ("4. Road Crash (SkyNews)", "test_images/skynews-car-crash-goodmayes_7250187.jpg", "road"),
        ("5. Pedestrian Incident", "test_images/Pedestrian-accident-3.jpg", "road"),
        ("6. Human Fall (Warm Indoor)", "test_images/fall_sample.jpg", "fall"),
        ("7. Normal Traffic (Busy)", "test_images/images (3).jpg", "normal"),
        ("8. Normal Traffic (Highway)", "test_images/images (4).jpg", "normal"),
        ("9. Normal Traffic (Street)", "test_images/images.jpg", "normal"),
        ("10. Hard Negative (Action)", hardneg_sample, "normal"),
    ]

    print("\n" + "=" * 110)
    print(f"{'Scenario':<30} | {'Accident':<8} | {'Type':<22} | {'Score':<8} | {'Scores (F, R, Fl)':<25} | {'Status'}")
    print("=" * 110)

    all_passed = True
    for desc, path, expected_cat in test_scenarios:
        if not path or not os.path.exists(path):
            print(f"{desc:<30} | SKIPPED (File not found)")
            continue

        with open(path, "rb") as f:
            img_bytes = f.read()

        t0 = time.time()
        resp = requests.post(PREDICT_URL, data=img_bytes, headers={"Content-Type": "image/jpeg"}, timeout=20)
        lat = (time.time() - t0) * 1000
        assert resp.status_code == 200, f"Expected status 200, got {resp.status_code}"
        d = resp.json()

        acc = d.get("accident", False)
        act_type = d.get("type", "non_accident")
        sc = d.get("score", 0.0)
        scores = d.get("scores", {})
        f_sc = scores.get("fire", 0.0)
        r_sc = scores.get("road", 0.0)
        fl_sc = scores.get("fall", 0.0)
        score_str = f"F:{f_sc:.2f} R:{r_sc:.2f} Fl:{fl_sc:.2f}"

        is_pass = False
        if expected_cat == "normal":
            is_pass = (acc == False)
        elif expected_cat == "fire":
            # Score must be neural detection (e.g. >= 0.40) and NOT 0.11 spatial area ratio
            is_pass = (acc == True and "fire" in act_type and sc >= 0.40 and abs(sc - 0.11) > 0.05)
        elif expected_cat == "fall":
            is_pass = (acc == True and "fall" in act_type and f_sc < 0.40)
        elif expected_cat == "road":
            is_pass = (acc == True and ("road" in act_type or "fall" in act_type))

        if not is_pass:
            all_passed = False

        status_str = "PASS" if is_pass else "FAIL"
        print(f"{desc:<30} | {str(acc):<8} | {act_type:<22} | {sc:<8.3f} | {score_str:<25} | {status_str}")

    print("=" * 110)
    print(f"Overall Live Space Result: {'ALL TESTS PASSED (100% SUCCESS)' if all_passed else 'SOME TESTS FAILED'}")
    assert all_passed, "Live Space verification failed!"

if __name__ == "__main__":
    check_space()
