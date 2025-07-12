
import os
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import base64
from datetime import datetime
from PIL import Image
from zoneinfo import ZoneInfo
import pandas as pd
import tempfile
import time
import subprocess

# Config
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🪖", layout="wide")

# Constants
MODEL_PATH = "models/best.onnx"
LABELS = ["NO Helmet", "ON. Helmet"]
ALARM_PATH = "assets/alarm.mp3"
SAVE_DIR = "violations"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load ONNX Model
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

session, input_name = load_model()

# Preprocess
def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Postprocess
def postprocess(outputs, threshold=0.5):
    predictions = outputs[0][0]
    results = []
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf > threshold:
            results.append((int(cls), float(conf), (int(x1), int(y1), int(x2), int(y2))))
    return results

# Alarm
def play_alarm():
    subprocess.Popen(["mpg123", ALARM_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Sidebar UI
st.sidebar.markdown("<h1 style='text-align:center;'>CapSure</h1>", unsafe_allow_html=True)
st.sidebar.markdown("🪖 Real-time Helmet Compliance Detection")
st.sidebar.markdown("---")

start_camera = st.sidebar.toggle("📷 Camera ON/OFF", value=False, key="cam_toggle")
reset_trigger = st.sidebar.button("🔁 RESET", use_container_width=True)

# Main Title
st.markdown("<h1 style='text-align:center; color:#3ABEFF;'>CapSure - Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# History init
if 'history' not in st.session_state:
    st.session_state.history = []

if 'violation' not in st.session_state:
    st.session_state.violation = False

# Detection loop
FRAME_WINDOW = st.empty()

if start_camera:
    cap = cv2.VideoCapture(0)
    st.info("🎥 Live camera started. Press RESET if alarm triggered.")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            img_input = preprocess(frame)
            outputs = session.run(None, {input_name: img_input})
            detections = postprocess(outputs)

            alert_triggered = False

            for cls_id, conf, (x1, y1, x2, y2) in detections:
                label = LABELS[cls_id]
                color = (0, 255, 0) if label == "ON. Helmet" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if label == "NO Helmet":
                    alert_triggered = True

            if alert_triggered and not st.session_state.violation:
                play_alarm()
                timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(SAVE_DIR, f"violation_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
                st.session_state.history.insert(0, {
                    "timestamp": timestamp,
                    "class": "NO Helmet",
                    "image_path": save_path
                })
                st.session_state.violation = True
                st.warning("🚨 Violation Detected! Please RESET to continue.")

            if st.session_state.violation:
                # Freeze frame after violation
                FRAME_WINDOW.image(frame, channels="BGR", caption="Violation detected.")
                break
            else:
                FRAME_WINDOW.image(frame, channels="BGR")

            if not start_camera:
                break
    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        cap.release()

# Reset functionality
if reset_trigger:
    st.session_state.violation = False
    st.rerun()

# Defect Log Page
st.markdown("---")
st.markdown("## 📋 Defect Log (Recent Violations)")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    df.index = df.index + 1
    st.dataframe(df)

    csv = df.to_csv(index=True).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "defect_log.csv", "text/csv")
else:
    st.info("No helmet violations recorded yet.")
