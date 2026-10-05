from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import os

app = FastAPI(title="Accident API - Fire + Email")

model = YOLO("yolov8n.pt")


EMAIL_ADDRESS = "projectg595@gmail.com"
APP_PASSWORD = "dqnuhcmfhxkeprxz"
TO_EMAIL = "projectg595@gmail.com"

STRONG_FIRE_RATIO = 0.040
MEDIUM_FIRE_RATIO = 0.026
MEDIUM_HITS_NEEDED = 2

medium_hits = 0


def send_email_alert(accident_type="fire_smoke_accident", score=0.0):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = "🚨 Accident Detected Alert"
    body = f"""Accident Detection Alert

Type: {accident_type}
Confidence Score: {score:.4f}
Time: {now}

Message:
Accident detected by ESP32-CAM AI monitoring system.
Please check immediately.
"""

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL

    msg.attach(MIMEText(body, "plain"))

    if os.path.exists("last_accident.jpg"):
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


@app.get("/")
def home():
    return {"status": "ok", "message": "Accident API is running"}


@app.post("/predict")
async def predict(request: Request):
    global medium_hits

    data = await request.body()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"accident": False, "error": "bad_image"}, status_code=400)

    cv2.imwrite("last_accident.jpg", frame)

    fire_ratio, strong_hit, medium_hit = is_fire_like(frame)
    
    if strong_hit:
        medium_hits = 0
        return {
            "accident": True,
            "type": "fire_smoke_accident",
            "score": float(fire_ratio)
        }

    if medium_hit:
        medium_hits += 1
    else:
        medium_hits = 0

    if medium_hits >= MEDIUM_HITS_NEEDED:
        medium_hits = 0
        return {
            "accident": True,
            "type": "fire_smoke_accident",
            "score": float(fire_ratio)
        }

    return {
        "accident": False,
        "type": "non_accident",
        "score": float(fire_ratio)
    }

@app.post("/send_email")
async def send_email(request: Request):
    try:
        data = await request.json()
        accident_type = data.get("type", "fire_smoke_accident")
        score = float(data.get("score", 0.0))

        send_email_alert(accident_type, score)

        return {
            "ok": True,
            "message": "Email sent",
            "type": accident_type,
            "score": score
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)