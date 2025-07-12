import os
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
import zipfile

# Configuration
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🪖", layout="wide")

# Constants
MODEL_ZIP = "best.zip"
MODEL_ONNX = "best.onnx"
LABELS = ["NO Helmet", "ON. Helmet"]

# Extract ONNX model if not already done
if not os.path.exists(MODEL_ONNX) and os.path.exists(MODEL_ZIP):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as z:
        z.extractall(".")

# Load model
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

session, input_name = load_model()

# Preprocess image
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)
    img_normalized = img_transposed.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

# Correct postprocess for YOLOv5
def postprocess(outputs, conf_threshold=0.3):
    predictions = outputs[0][0]  # (8400, 85)
    boxes = []
    for pred in predictions:
        x_center, y_center, width, height = pred[0:4]
        objectness = pred[4]
        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        confidence = objectness * class_conf

        if confidence > conf_threshold:
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            boxes.append((class_id, float(confidence), (x1, y1, x2, y2)))
    return boxes

# State setup
if "history" not in st.session_state:
    st.session_state.history = []
if "violated" not in st.session_state:
    st.session_state.violated = False

# UI - Sidebar
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.markdown(
    "<h1 style='text-align:center; color:yellow;'>CapSure</h1><h2 style='text-align:center; color:yellow;'>Helmet Detection</h2>",
    unsafe_allow_html=True
)
start = st.sidebar.toggle("📷 Camera ON/OFF")
if st.sidebar.button("🔁 RESET"):
    st.session_state.violated = False

# UI - Main
st.title("🪖 Helmet Compliance Detection")

if start and not st.session_state.violated:
    img_file = st.camera_input("📸 Capture Image")
    if img_file:
        # Read & convert
        img_pil = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Detection
        inp = preprocess(frame)
        outs = session.run(None, {input_name: inp})
        det = postprocess(outs)

        alert = False
        for clsid, conf, (x1, y1, x2, y2) in det:
            label = LABELS[clsid]
            color = (0, 255, 0) if clsid == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if clsid == 0:
                alert = True

        # Display results
        st.image(frame, channels="BGR", use_column_width=True)

        if alert:
            st.warning("🚨 Helmet Violation Detected!")
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            ts = now.strftime("%I:%M:%S %p @ %d %B, %Y")
            fn = f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            _, buffer = cv2.imencode(".jpg", frame)

            # Log violation
            st.session_state.history.insert(0, {
                "ts": ts,
                "class": "NO Helmet",
                "bytes": buffer.tobytes(),
                "fn": fn
            })
            st.session_state.violated = True

            st.download_button("⬇️ Download Snapshot", buffer.tobytes(), file_name=fn, mime="image/jpeg")
    elif st.session_state.violated:
        st.warning("⚠️ Detection paused due to previous violation. Click RESET to continue.")

# Violation Log
st.markdown("---")
st.subheader("📋 Violation Log")

if st.session_state.history:
    for i, entry in enumerate(st.session_state.history):
        cols = st.columns([3, 2, 1])
        with cols[0]:
            st.markdown(f"🕒 **Time:** {entry['ts']}")
        with cols[1]:
            st.markdown(f"🚧 **Class:** {entry['class']}")
        with cols[2]:
            st.download_button("Download", entry["bytes"], file_name=entry["fn"], mime="image/jpeg", key=f"dl_{i}")
else:
    st.info("✅ No violations recorded yet.")
