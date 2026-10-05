# Universal Golden Minute Emergency Response System (ML Component)

An automated multi-hazard AI emergency detection and life-saving dispatch platform designed for the critical "Golden Minute" window in trauma and accident response.

---

## 🌟 Overview & Architecture

The system ingests live camera video streams (webcam, CCTV, or ESP32-CAM) and evaluates frames across three synchronized hazard detection engines:

```
                          ┌───────────────────────────┐
                          │   Camera / Video Stream   │
                          └─────────────┬─────────────┘
                                        │ HTTP POST /predict
                                        ▼
                          ┌───────────────────────────┐
                          │   Flask API (app_final)   │
                          └─────────────┬─────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────┐               ┌───────────────┐               ┌───────────────┐
│ Road Accident │               │ Fall Detector │               │ Fire & Smoke  │
│ (YOLOv8n AI)  │               │ (YOLOv8n AI)  │               │ (HSV+RGB CV)  │
│road_expanded  │               │fall_expanded  │               │ Physics-based │
└───────┬───────┘               └───────┬───────┘               └───────┬───────┘
        │                               │                               │
        └───────────────────────────────┼───────────────────────────────┘
                                        │ Confirmed Emergency
                                        ▼
                        ┌───────────────────────────────┐
                        │   Emergency Dispatch Engine   │
                        │ - GPS Map Location Link       │
                        │ - High-Priority SSL Email     │
                        │ - Image Snapshot Attachment   │
                        └───────────────────────────────┘
```

---

## 🚦 Hazard Detection Engines

### 1. 🚗 Road Accident Detection (`road_expanded_best.pt`)
* **Model**: YOLOv8n fine-tuned on 26,250 balanced road images across 85,594 annotations.
* **Classes (4 Classes)**:
  * `0`: `human_incident` (Pedestrian casualties / struck victims)
  * `1`: `human_normal` (Normal pedestrians)
  * `2`: `vehicle_incident` (Vehicle collisions & crashes)
  * `3`: `vehicle_normal` (Normal cruising vehicles)
* **Performance on 10,646 Test Images**:
  * **`human_incident` Recall**: **89.22%** (+3.00% absolute increase over baseline).
  * **`vehicle_incident` Precision**: **92.03%** (+1.75% increase over baseline).
  * **`vehicle_incident` mAP@50**: **95.79%** (+1.20% increase over baseline).
  * **Latency**: ~24 ms (~41 FPS) on RTX 3050 GPU.

### 2. 🧍 Human Fall Detection (`fall_expanded_best.pt`)
* **Model**: YOLOv8n trained on multi-source fall datasets.
* **Sensitivity**: **86.6% Confidence** on true fall casualties with zero false alarms on normal standing/walking.
* **Latency**: ~20 ms (~49 FPS).

### 3. 🔥 Fire Detection (Physics-Based HSV + RGB Engine)
* **Algorithm**: Multi-space combustion color model ($R > G > B$, $R - B \ge 70$, $Cr - Cb \ge 38$) with incandescent core verification and blob aspect ratio filtering ($w/h \le 3.0$).
* **Reliability**: Eliminates false alarms on sunset skies, red cars, and red buses while maintaining 100% sensitivity on real flames.
* **Latency**: ~9 ms (>100 FPS), 0 MB GPU VRAM overhead.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Main Application
```bash
python app_final.py
```
Server starts on `http://localhost:5000`.

### 3. API Endpoints

| Endpoint | Method | Payload | Description |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | — | System health check |
| `/predict` | `POST` | Raw image bytes (`application/octet-stream`) | Multi-hazard emergency detection |
| `/gps_update` | `POST` | JSON `{"latitude": float, "longitude": float}` | Updates current GPS coordinates |
| `/gps_status` | `GET` | — | Returns latest stored GPS position |

---

## 🧪 Automated Integration Tests

All modules include dedicated regression test suites (20/20 tests passing):

```bash
# Run Road Accident Tests (7 Tests)
python test_app_road.py

# Run Human Fall Tests (5 Tests)
python test_app_fall.py

# Run Fire Detection Tests (8 Tests)
python test_app_fire.py
```

---

## 📁 Repository Structure

```
├── app_final.py                 # Primary multi-hazard Flask backend
├── app_fire.py                  # Standalone fire detection server
├── app_fall.py                  # Standalone fall detection server
├── road_expanded_best.pt        # Active fine-tuned road accident model (5.96 MB)
├── road_best.pt                 # Untouched baseline road accident model (17.5 MB)
├── fall_expanded_best.pt        # Active fall detection model (5.94 MB)
├── fall_accident_model_best.pt  # Untouched baseline fall model (5.94 MB)
├── test_app_road.py             # Road accident integration test suite
├── test_app_fall.py             # Fall detection integration test suite
├── test_app_fire.py             # Fire detection integration test suite
├── test_images/                 # Real-world test sample images
│   ├── fire_samples/            # Verified fire test images
│   └── ...                      # Road and street test scenes
├── conversation_summary.md      # Comprehensive technical & training report
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```
