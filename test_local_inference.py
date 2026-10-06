import json
import os
from app_final import app

client = app.test_client()

test_images = [
    ("Fire Basket", "test_images/fire_samples/fire_basket.jpg"),
    ("Pan Fire", "test_images/fire_samples/pan_fire.jpg"),
    ("Fall Sample", "test_images/fall_sample.jpg"),
    ("Car Crash SkyNews", "test_images/skynews-car-crash-goodmayes_7250187.jpg"),
    ("Pedestrian Incident", "test_images/Pedestrian-accident-3.jpg"),
    ("Normal Traffic 3", "test_images/images (3).jpg"),
    ("Normal Traffic 4", "test_images/images (4).jpg"),
]

print(f"{'Image Description':<25} | {'Accident':<8} | {'Type':<24} | {'Score':<8} | {'Scores (F,R,Fl)':<22}")
print("-" * 100)

for desc, path in test_images:
    if not os.path.exists(path):
        continue
    with open(path, "rb") as f:
        img_bytes = f.read()
    res = client.post("/predict", data=img_bytes, content_type="image/jpeg")
    d = res.get_json()
    sc = d.get("scores", {})
    f_sc = sc.get("fire", 0)
    r_sc = sc.get("road", 0)
    fl_sc = sc.get("fall", 0)
    sc_str = f"F:{f_sc:.2f} R:{r_sc:.2f} Fl:{fl_sc:.2f}"
    print(f"{desc:<25} | {str(d.get('accident')):<8} | {d.get('type', ''):<24} | {d.get('score', 0):<8.3f} | {sc_str:<22}")
