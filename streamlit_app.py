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
from PIL import Image

# Streamlit page config
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🪖", layout="wide")

# Constants
MODEL_ZIP_PATH = "best.zip"
MODEL_EXTRACTED_PATH = "best.onnx"
MODEL_PATH = "best.onnx"
LOGO_PATH = "logo.png"
LABELS = ["NO Helmet", "ON. Helmet"]

# Unzip model if not already extracted
if not os.path.exists(MODEL_EXTRACTED_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")

# Load ONNX model
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

session, input_name = load_model()

# Preprocess image for ONNX model
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)
    img_normalized = img_transposed.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

# Postprocess predictions
def postprocess(outputs, threshold=0.3):
    predictions = outputs[0][0]
    results = []
    for pred in predictions:
        if len(pred) < 6:
            continue
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf > threshold:
            results.append((int(cls), float(conf), (int(x1), int(y1), int(x2), int(y2))))
    return results

# Fake alarm (for Streamlit Cloud)
def play_alarm():
    st.warning("🚨 Violation detected! (Sound not supported on cloud)")

# Sidebar UI
st.sidebar.image(LOGO_PATH, use_container_width=True)
st.sidebar.markdown(
    """
    <h1 style='text-align:center; color:yellow; font-size: 36px;'>CapSure</h1>
    <h2 style='text-align:center; color:yellow; font-size: 20px;'>Real-time Helmet Compliance Detection</h2>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
reset_trigger = st.sidebar.button("🔁 RESET", use_container_width=True)

# Title
st.markdown("<h1 style='text-align:center; color:#3ABEFF;'>CapSure - Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Init session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'violation' not in st.session_state:
    st.session_state.violation = False

# CAMERA INPUT UI
img_file = st.camera_input("📷 Capture Image")

if img_file and not st.session_state.violation:
    image = Image.open(img_file)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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

    if alert_triggered:
        play_alarm()
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        formatted_time = now.strftime("%I:%M:%S %p @ %d %B, %Y")
        filename = f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        st.session_state.history.insert(0, {
            "timestamp": formatted_time,
            "class": "NO Helmet",
            "filename": filename,
            "image_bytes": img_bytes
        })

        st.session_state.violation = True
        st.warning("🚨 Violation Detected! Please RESET to continue.")

        st.download_button("⬇️ Download Violation Snapshot", img_bytes, filename, "image/jpeg")

    st.image(frame, channels="BGR", use_container_width=True)

elif st.session_state.violation:
    st.warning("❗ Detection paused. Press RESET to continue.")

# RESET button
if reset_trigger:
    st.session_state.violation = False
    st.rerun()

# DEFECT LOG
st.markdown("---")
st.markdown("## 📋 Defect Log (Recent Violations)")

if st.session_state.history:
    for i, entry in enumerate(st.session_state.history):
        cols = st.columns([2, 2, 1])
        with cols[0]:
            st.markdown(f"**🕒 Time:** {entry['timestamp']}")
        with cols[1]:
            st.markdown(f"**🚧 Class:** {entry['class']}")
        with cols[2]:
            st.download_button("Download Image", data=entry["image_bytes"],
                               file_name=entry["filename"], mime="image/jpeg",
                               key=f"download_{i}")
else:
    st.info("No helmet violations recorded yet.")
