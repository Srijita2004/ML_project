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

app = Flask(__name__)

# Load trained fall detection model (prefer new improved model)
MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "models", "production", "fall_v2_hardneg_best.pt"),
    os.path.join(os.path.dirname(__file__), "fall_expanded_best.pt"),
    os.path.join(os.path.dirname(__file__), "runs", "fall_subset5k_v1", "weights", "best.pt"),
    os.path.join(os.path.dirname(__file__), "best.pt"),
    os.path.join(os.path.dirname(__file__), "fall_accident_model_best.pt")
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), "best.pt")
print(f"[MODEL] Loading fall detection model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

EMAIL_ADDRESS = "projectg595@gmail.com"
APP_PASSWORD = "dqnuhcmfhxkeprxz"   
TO_EMAIL = "projectg595@gmail.com"


def send_email_alert(accident_type="human_fall_accident", score=1.0):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = "🚨 Fall Accident Detected Alert"
    body = f"""Accident Detection Alert

Type: {accident_type}
Confidence Score: {score:.4f}
Time: {now}

Message:
Human fall detected by ESP32-CAM AI monitoring system.
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


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Fall API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ESP32 raw JPEG body পাঠাচ্ছে
        file = request.data

        if not file:
            return jsonify({
                "accident": False,
                "type": "non_accident",
                "score": 0.0,
                "result": "normal",
                "error": "empty_body"
            }), 400

        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "accident": False,
                "type": "non_accident",
                "score": 0.0,
                "result": "normal",
                "error": "bad_image"
            }), 400

        # Save last received image
        cv2.imwrite("last_accident.jpg", img)

        results = model(img, verbose=False)

        fall_detected = False
        best_conf = 0.0

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                fall_detected = True
                if r.boxes.conf is not None and len(r.boxes.conf) > 0:
                    confs = r.boxes.conf.cpu().numpy().tolist()
                    best_conf = max(confs) if confs else 1.0
                else:
                    best_conf = 1.0

        if fall_detected:
            return jsonify({
                "emergency": True,
                "type": "fall",
                "confidence": float(best_conf),
                "accident": True,
                "score": float(best_conf),
                "result": "fall detected"
            })
        else:
            return jsonify({
                "emergency": False,
                "type": "none",
                "confidence": 0.0,
                "accident": False,
                "score": 0.0,
                "result": "normal"
            })

    except Exception as e:
        return jsonify({
            "accident": False,
            "type": "non_accident",
            "score": 0.0,
            "result": "normal",
            "error": str(e)
        }), 500


@app.route("/send_email", methods=["POST"])
def send_email():
    try:
        data = request.get_json(force=True)
        accident_type = data.get("type", "human_fall_accident")
        score = float(data.get("score", 1.0))

        send_email_alert(accident_type, score)

        return jsonify({
            "ok": True,
            "message": "Email sent",
            "type": accident_type,
            "score": score
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)