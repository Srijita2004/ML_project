import json
import os
import cv2
from app_final import app

client = app.test_client()

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
                    if line.strip().startswith("1 "): # Class 1 = Smoke
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
    ("1. Obvious Fire", "test_images/fire_samples/fire_basket.jpg", "fire"),
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

print("=" * 110)
print(f"{'Scenario':<28} | {'Accident':<8} | {'Type':<22} | {'Score':<8} | {'Scores (F, R, Fl)':<25} | {'Status'}")
print("=" * 110)

all_passed = True

for desc, path, expected_cat in test_scenarios:
    if not path or not os.path.exists(path):
        print(f"{desc:<28} | SKIPPED (File not found)")
        continue

    with open(path, "rb") as f:
        img_bytes = f.read()

    res = client.post("/predict", data=img_bytes, content_type="image/jpeg")
    assert res.status_code == 200, f"Expected status 200, got {res.status_code}"
    d = res.get_json()

    acc = d.get("accident", False)
    act_type = d.get("type", "non_accident")
    sc = d.get("score", 0.0)
    scores = d.get("scores", {})
    f_sc = scores.get("fire", 0.0)
    r_sc = scores.get("road", 0.0)
    fl_sc = scores.get("fall", 0.0)

    score_str = f"F:{f_sc:.2f} R:{r_sc:.2f} Fl:{fl_sc:.2f}"

    # Validation criteria:
    # - If expected normal: accident must be False
    # - If expected fire: accident must be True, type must contain fire, score must NOT be ~0.11 pixel area ratio
    # - If expected fall: accident must be True, type must be fall, must NOT be hijacked by fire
    # - If expected road: accident must be True, type must be road or fall/human incident
    is_pass = False
    if expected_cat == "normal":
        is_pass = (acc == False)
    elif expected_cat == "fire":
        is_pass = (acc == True and "fire" in act_type and sc >= 0.40)
    elif expected_cat == "fall":
        is_pass = (acc == True and "fall" in act_type and f_sc < 0.40)
    elif expected_cat == "road":
        is_pass = (acc == True and ("road" in act_type or "fall" in act_type))

    if not is_pass:
        all_passed = False

    status_str = "PASS" if is_pass else "FAIL"
    print(f"{desc:<28} | {str(acc):<8} | {act_type:<22} | {sc:<8.3f} | {score_str:<25} | {status_str}")

print("=" * 110)
print(f"Overall Phase 3 Regression Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
assert all_passed, "Regression test failed!"
