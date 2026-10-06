from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import os
import time

app = Flask(__name__)

# =====================================================
# GPS STORAGE
# =====================================================
latest_latitude  = None
latest_longitude = None

# =====================================================
# EMAIL COOLDOWN — camera r pulse er jonno ALAG
# =====================================================
last_email_time_camera = 0
last_email_time_pulse  = 0
EMAIL_COOLDOWN         = 35

# =====================================================
# MODELS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

fall_model_candidates = [
    os.path.join(BASE_DIR, "models", "production", "fall_v2_hardneg_best.pt"),
    os.path.join(BASE_DIR, "fall_expanded_best.pt"),
    os.path.join(BASE_DIR, "fall_accident_model_best.pt"),
    os.path.join(BASE_DIR, "models", "rollback", "fall_expanded_best_v1.pt"),
]
fall_model_path = next((p for p in fall_model_candidates if os.path.exists(p)), os.path.join(BASE_DIR, "fall_expanded_best.pt"))
print(f"[MODEL] Loading fall model: {fall_model_path}")
fall_model = YOLO(fall_model_path)

road_model_candidates = [
    os.path.join(BASE_DIR, "models", "production", "road_v2_cctv_best.pt"),
    os.path.join(BASE_DIR, "road_expanded_best.pt"),
    os.path.join(BASE_DIR, "road_best.pt"),
    os.path.join(BASE_DIR, "models", "rollback", "road_expanded_best_v1.pt"),
]
road_model_path = next((p for p in road_model_candidates if os.path.exists(p)), os.path.join(BASE_DIR, "road_expanded_best.pt"))
print(f"[MODEL] Loading road model: {road_model_path}")
road_model = YOLO(road_model_path)

# =====================================================
# EMAIL CONFIGURATION (Supports ENV overrides)
# =====================================================
EMAIL_ADDRESS = os.environ.get("ALERT_EMAIL_ADDRESS", "projectg595@gmail.com")
APP_PASSWORD  = os.environ.get("ALERT_APP_PASSWORD", "dqnuhcmfhxkeprxz")
TO_EMAIL      = os.environ.get("ALERT_TO_EMAIL", "projectg595@gmail.com")

# =====================================================
# FIRE SETTINGS
# =====================================================
STRONG_FIRE_RATIO  = 0.040
MEDIUM_FIRE_RATIO  = 0.026

# =====================================================
# ROAD SETTINGS (Calibrated on Validation CCTV Set)
# =====================================================
ACCIDENT_CLASSES   = ["human_incident", "vehicle_incident"]
ROAD_CONF_THRES    = 0.35

# =====================================================
# FALL SETTINGS (Calibrated with Hard Negative Rejection)
# =====================================================
FALL_CONF_THRES     = 0.45
FALL_MIN_AREA_RATIO = 0.01

# =====================================================
# SEND EMAIL
# =====================================================
def send_email_alert(
    accident_type="accident",
    score=0.0,
    latitude=None,
    longitude=None,
    heart_rate=None,
    source="CAMERA"
):
    global last_email_time_camera, last_email_time_pulse

    now_time = time.time()

    # Camera r pulse er jonno alag cooldown check
    if source == "PULSE_SENSOR":
        if now_time - last_email_time_pulse < EMAIL_COOLDOWN:
            print(f"[PULSE] Cooldown active, skipping...")
            return
        last_email_time_pulse = now_time
    else:
        if now_time - last_email_time_camera < EMAIL_COOLDOWN:
            print(f"[CAMERA] Cooldown active, skipping...")
            return
        last_email_time_camera = now_time

    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        map_link = "N/A"
        if latitude is not None and longitude is not None:
            try:
                map_link = f"https://maps.google.com/?q={float(latitude)},{float(longitude)}"
            except:
                map_link = "N/A"

        hr_display  = f"{heart_rate} BPM" if heart_rate else "N/A"
        lat_display = str(latitude)  if latitude  is not None else "N/A"
        lng_display = str(longitude) if longitude is not None else "N/A"

        subject = "🚨 Accident Detected Alert"
        body = f"""
SMART ACCIDENT DETECTION SYSTEM
====================================
Detection Source  : {source}
Accident Type     : {accident_type}
Confidence Score  : {score:.4f}
Heart Rate        : {hr_display}
Latitude          : {lat_display}
Longitude         : {lng_display}
Google Maps       : {map_link}
Date & Time       : {now}
====================================
PLEASE CHECK IMMEDIATELY
"""
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"]    = EMAIL_ADDRESS
        msg["To"]      = TO_EMAIL
        msg.attach(MIMEText(body, "plain"))

        # Camera hole photo attach koro
        if source != "PULSE_SENSOR" and os.path.exists("last_accident.jpg"):
            with open("last_accident.jpg", "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                "attachment; filename=accident.jpg"
            )
            msg.attach(part)

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(EMAIL_ADDRESS, APP_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
        server.quit()
        print(f"✅ EMAIL SENT! [{source}] → {accident_type}")

    except Exception as e:
        print(f"❌ EMAIL ERROR [{source}]:", e)

# =====================================================
# FIRE DETECTION
# =====================================================
def is_fire_like(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if h == 0 or w == 0:
        return 0.0, False, False

    # Fast scaling to max 640 dim for sub-10ms real-time execution
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

    # 1. Color spaces
    hsv = cv2.cvtColor(frame_eval, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = cv2.split(hsv)
    b, g, r = cv2.split(frame_eval)
    ycbcr = cv2.cvtColor(frame_eval, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    # 2. Physics-based Flame Color Rules (HSV + RGB + YCrCb)
    flame_mask = (
        ((h_c <= 28) | (h_c >= 165)) &
        (s_c >= 95) & (v_c >= 155) &
        (r >= 180) & (r > g) & (g > b) &
        ((r.astype(int) - b.astype(int)) >= 70) &
        ((cr.astype(int) - cb.astype(int)) >= 38)
    ).astype(np.uint8) * 255

    # 3. Luminous Incandescent Core (Combustion emission check)
    core_mask = (
        (v_c >= 215) & (r >= 215) & (g >= 120) &
        ((r.astype(int) - b.astype(int)) >= 85)
    )

    # 4. Morphological Noise Rejection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(flame_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # 5. Connected Component / Blob Geometry Analysis
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(50, int(tot_pixels * 0.0003))

    valid_fire_pixels = 0
    max_blob_area = 0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        
        # Reject horizontal vehicle stripes / barriers (width > 3x height)
        x, y_box, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(1, ch)
        if aspect > 3.0:
            continue
        
        # Verify incandescent flame core presence inside this blob
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

    strong_hit = (fire_ratio >= STRONG_FIRE_RATIO) or (max_blob_ratio >= 0.035)
    medium_hit = (fire_ratio >= MEDIUM_FIRE_RATIO) or (max_blob_ratio >= 0.018)

    return fire_ratio, strong_hit, medium_hit

# =====================================================
# CORS SUPPORT
# =====================================================
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


# =====================================================
# HOME
# =====================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Combined Accident API is running"})

# =====================================================
# GPS UPDATE
# =====================================================
@app.route("/gps_update", methods=["POST"])
def gps_update():
    global latest_latitude, latest_longitude
    try:
        data = request.get_json(force=True)
        latest_latitude  = data.get("latitude")
        latest_longitude = data.get("longitude")
        print(f"📍 GPS: {latest_latitude}, {latest_longitude}")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# =====================================================
# GPS STATUS
# =====================================================
@app.route("/gps_status", methods=["GET"])
def gps_status():
    return jsonify({
        "latitude":  latest_latitude,
        "longitude": latest_longitude
    })

# =====================================================
# PREDICT
# =====================================================
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        file = request.data
        if (not file or len(file) == 0) and request.files:
            file_obj = request.files.get("file") or request.files.get("image")
            if file_obj:
                file = file_obj.read()

        if not file:
            return jsonify({
                "accident": False,
                "emergency": False,
                "type": "non_accident",
                "score": 0.0,
                "confidence": 0.0,
                "result": "normal",
                "error": "empty_body"
            }), 400

        npimg = np.frombuffer(file, np.uint8)
        img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({
                "accident": False,
                "emergency": False,
                "type": "non_accident",
                "score": 0.0,
                "confidence": 0.0,
                "result": "normal",
                "error": "bad_image"
            }), 400

        h, w = img.shape[:2]
        frame_area = h * w if h > 0 and w > 0 else 1

        def save_snapshot(frame):
            try:
                cv2.imwrite("last_accident.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            except Exception:
                pass

        # 1. FIRE EVALUATION
        fire_ratio, strong_hit, medium_hit = is_fire_like(img)
        if strong_hit:
            save_snapshot(img)
            return jsonify({
                "accident": True,
                "emergency": True,
                "type": "fire_smoke_accident",
                "score": float(fire_ratio),
                "confidence": float(fire_ratio),
                "result": "fire detected",
                "labels": ["fire_combustion"]
            })

        # 2. RUN ROAD MODEL FIRST
        road_results = road_model(img, conf=0.30, verbose=False)

        detected_labels = []
        best_road_conf = 0.0
        best_road_label = None

        for r in road_results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].item()) if r.boxes.cls is not None else -1
                conf = float(r.boxes.conf[i].item()) if r.boxes.conf is not None else 0.0
                cname = road_model.names.get(cls_id, str(cls_id))
                detected_labels.append(cname)
                if cname in ACCIDENT_CLASSES and conf >= ROAD_CONF_THRES:
                    if conf > best_road_conf:
                        best_road_conf = conf
                        best_road_label = cname

        # Fast-path: Unequivocal high-confidence road collision or pedestrian incident (>= 0.70)
        if best_road_conf >= 0.70:
            save_snapshot(img)
            return jsonify({
                "accident": True,
                "emergency": True,
                "type": "road_vehicle_accident",
                "score": float(best_road_conf),
                "confidence": float(best_road_conf),
                "result": "road accident detected",
                "labels": detected_labels
            })

        # 3. RUN FALL MODEL IF NO HIGH-CONFIDENCE ROAD COLLISION
        fall_results = fall_model(img, conf=0.30, verbose=False)
        best_fall_conf = 0.0
        for r in fall_results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                conf = float(r.boxes.conf[i].item()) if r.boxes.conf is not None else 0.0
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                area_ratio = box_area / frame_area
                if conf >= FALL_CONF_THRES and area_ratio >= FALL_MIN_AREA_RATIO:
                    if conf > best_fall_conf:
                        best_fall_conf = conf
                        if "Fall-Detected" not in detected_labels:
                            detected_labels.append("Fall-Detected")

        # 3. SYNTHESIZE MULTI-MODAL ACCIDENT OUTCOME
        # Case A: Pedestrian road incident detected
        if best_road_label == "human_incident" and best_road_conf >= ROAD_CONF_THRES:
            save_snapshot(img)
            return jsonify({
                "accident": True,
                "emergency": True,
                "type": "road_vehicle_accident",
                "score": float(best_road_conf),
                "confidence": float(best_road_conf),
                "result": "road accident detected",
                "labels": detected_labels
            })

        # Case B: Fall is detected with higher confidence than vehicle incident
        if best_fall_conf >= FALL_CONF_THRES and best_fall_conf >= best_road_conf:
            save_snapshot(img)
            return jsonify({
                "accident": True,
                "emergency": True,
                "type": "human_fall_accident",
                "score": float(best_fall_conf),
                "confidence": float(best_fall_conf),
                "result": "fall detected",
                "labels": detected_labels
            })

        # Case C: Vehicle crash detected
        if best_road_conf >= ROAD_CONF_THRES:
            save_snapshot(img)
            return jsonify({
                "accident": True,
                "emergency": True,
                "type": "road_vehicle_accident",
                "score": float(best_road_conf),
                "confidence": float(best_road_conf),
                "result": "road accident detected",
                "labels": detected_labels
            })

        # Case D: Fall detected without road incident
        if best_fall_conf >= FALL_CONF_THRES:
            save_snapshot(img)
            return jsonify({
                "accident": True,
                "emergency": True,
                "type": "human_fall_accident",
                "score": float(best_fall_conf),
                "confidence": float(best_fall_conf),
                "result": "fall detected",
                "labels": detected_labels
            })

        # Case E: Normal Scene (Non-Accident)
        return jsonify({
            "accident": False,
            "emergency": False,
            "type": "non_accident",
            "score": 0.0,
            "confidence": 0.0,
            "result": "normal",
            "labels": detected_labels
        })

    except Exception as e:
        return jsonify({
            "accident": False,
            "emergency": False,
            "type": "non_accident",
            "score": 0.0,
            "confidence": 0.0,
            "result": "normal",
            "error": str(e)
        }), 500

# =====================================================
# SEND EMAIL — ESP32-CAM theke (30 sec por call kore)
# =====================================================
@app.route("/send_email", methods=["POST"])
def send_email():
    try:
        data          = request.get_json(force=True)
        accident_type = data.get("type", "accident")
        score         = float(data.get("score", 0.0))

        print(f"📧 Camera email: {accident_type}")
        print(f"📍 GPS: {latest_latitude}, {latest_longitude}")

        send_email_alert(
            accident_type=accident_type,
            score=score,
            latitude=latest_latitude,
            longitude=latest_longitude,
            source="ESP32-CAM"
        )
        return jsonify({"ok": True, "message": "Email sent"})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# =====================================================
# PULSE ALERT — ESP32 DevKit theke
# =====================================================
@app.route("/pulse_alert", methods=["POST"])
def pulse_alert():
    try:
        data       = request.get_json(force=True)
        heart_rate = data.get("heart_rate")
        latitude   = data.get("latitude")
        longitude  = data.get("longitude")

        lat = latitude  if latitude  is not None else latest_latitude
        lng = longitude if longitude is not None else latest_longitude

        print(f"💓 Pulse alert! BPM: {heart_rate}")
        print(f"📍 Location: {lat}, {lng}")

        send_email_alert(
            accident_type="abnormal_heart_rate",
            score=1.0,
            heart_rate=heart_rate,
            latitude=lat,
            longitude=lng,
            source="PULSE_SENSOR"
        )
        return jsonify({"ok": True})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port)