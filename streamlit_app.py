import os
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import base64
import zipfile

# ──────────────────────────────
# CONFIGURATION
# ──────────────────────────────
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🫶", layout="wide")

MODEL_ZIP = "best.zip"
MODEL_PATH = "best.onnx"
LOGO_PATH = "logo.png"
ALARM_PATH = "alarm.mp3"
LABELS = ["NO Helmet", "ON. Helmet"]  # Adjust if class order is different in your model

# ──────────────────────────────
# MODEL EXTRACTION (IF NEEDED)
# ──────────────────────────────
if not os.path.exists(MODEL_PATH) and os.path.exists(MODEL_ZIP):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as z:
        z.extractall(".")

# ──────────────────────────────
# LOAD MODEL
# ──────────────────────────────
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

session, input_name = load_model()

# ──────────────────────────────
# PREPROCESS
# ──────────────────────────────
def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ──────────────────────────────
# POSTPROCESS
# ──────────────────────────────
def postprocess(outputs, threshold=0.3):
    predictions = outputs[0][0]
    results = []
    for pred in predictions:
        if len(pred) >= 6:
            x1, y1, x2, y2, conf, cls = pred[:6]
            if conf > threshold:
                results.append((int(cls), float(conf), (int(x1), int(y1), int(x2), int(y2))))
    return results

# ──────────────────────────────
# ALARM (OPTIONAL)
# ──────────────────────────────
def play_alarm():
    st.warning("🚨 Violation detected! (Sound disabled on web UI)")

# ──────────────────────────────
# SIDEBAR UI
# ──────────────────────────────
st.sidebar.image(LOGO_PATH, use_column_width=True)
st.sidebar.markdown("""
    <h1 style='text-align:center; color:yellow;'>CapSure</h1>
    <h3 style='text-align:center; color:yellow;'>Helmet Detection</h3>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

start_camera = st.sidebar.toggle("📷 Camera ON/OFF", value=False)
reset_trigger = st.sidebar.button("🔁 RESET", use_container_width=True)

# ──────────────────────────────
# MAIN UI HEADER
# ──────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#3ABEFF;'>CapSure - Helmet Detection System</h1>
---
""", unsafe_allow_html=True)

# Session init
if 'history' not in st.session_state:
    st.session_state.history = []
if 'violation' not in st.session_state:
    st.session_state.violation = False

frame_placeholder = st.empty()

# ──────────────────────────────
# DETECTION LOGIC
# ──────────────────────────────
if start_camera:
    cap = cv2.VideoCapture(0)
    st.info("🎥 Live camera started. Press RESET to clear violation.")
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
                label = LABELS[cls_id] if cls_id < len(LABELS) else f"Class {cls_id}"
                color = (0, 255, 0) if label == "ON. Helmet" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if label == "NO Helmet":
                    alert_triggered = True

            # If violation detected
            if alert_triggered and not st.session_state.violation:
                play_alarm()
                now = datetime.now(ZoneInfo("Asia/Kolkata"))
                timestamp = now.strftime("%I:%M:%S %p @ %d %B, %Y")
                filename = f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()

                st.session_state.history.insert(0, {
                    "timestamp": timestamp,
                    "class": "NO Helmet",
                    "filename": filename,
                    "image_bytes": img_bytes
                })

                st.session_state.violation = True
                st.warning("🚨 Helmet Violation Detected!")
                st.download_button("⬇️ Download Snapshot", img_bytes, filename, "image/jpeg")

            # Display frame
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            if not start_camera or st.session_state.violation:
                break

    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        cap.release()

# ──────────────────────────────
# RESET LOGIC
# ──────────────────────────────
if reset_trigger:
    st.session_state.violation = False
    st.rerun()

# ──────────────────────────────
# VIOLATION LOG DISPLAY
# ──────────────────────────────
st.markdown("---")
st.subheader("📋 Violation Log")
if st.session_state.history:
    df = pd.DataFrame([{"Timestamp": h["timestamp"], "Class": h["class"]} for h in st.session_state.history])
    df.index += 1
    st.dataframe(df)

    csv = df.to_csv(index=True).encode('utf-8')
    st.download_button("⬇️ Download Log", csv, "violation_log.csv", "text/csv")
else:
    st.info("✅ No helmet violations recorded yet.")
