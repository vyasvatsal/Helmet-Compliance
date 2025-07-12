import os
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import zipfile

# ---------------------- App Config ----------------------
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🪖", layout="wide")

# ---------------------- Constants ----------------------
MODEL_ZIP = "best.zip"
MODEL_ONNX = "best.onnx"
LABELS = ["ON. Helmet", "NO Helmet"]
LOGO_PATH = "logo.png"

# ---------------------- Unzip Model If Needed ----------------------
if not os.path.exists(MODEL_ONNX) and os.path.exists(MODEL_ZIP):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

session, input_name = load_model()

# ---------------------- Preprocess ----------------------
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)
    img_normalized = img_transposed.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

# ---------------------- Postprocess ----------------------
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

# ---------------------- Session Init ----------------------
if 'history' not in st.session_state:
    st.session_state.history = []

if 'violation' not in st.session_state:
    st.session_state.violation = False

# ---------------------- Sidebar ----------------------
st.sidebar.image(LOGO_PATH, use_container_width=True)
st.sidebar.markdown(
    """
    <h1 style='text-align:center; color:yellow;'>CapSure</h1>
    <h2 style='text-align:center; color:yellow;'>Helmet Detection</h2>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

start_camera = st.sidebar.toggle("📷 Camera ON/OFF", value=False, key="cam_toggle")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.2, 0.7, 0.3, 0.05)
reset_trigger = st.sidebar.button("🔁 RESET", use_container_width=True)

# ---------------------- Main Title ----------------------
st.markdown("<h1 style='text-align:center; color:#3ABEFF;'>🪖 CapSure - Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

frame_placeholder = st.empty()

# ---------------------- Detection Loop ----------------------
if start_camera:
    cap = cv2.VideoCapture(0)
    st.info("🎥 Live camera started. Press RESET to detect again after a violation.")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            img_input = preprocess(frame)
            outputs = session.run(None, {input_name: img_input})
            detections = postprocess(outputs, threshold=conf_thresh)

            alert_triggered = False

            for cls_id, conf, (x1, y1, x2, y2) in detections:
                label = LABELS[cls_id] if cls_id < len(LABELS) else f"Class {cls_id}"
                if cls_id == 0:
                    color = (0, 255, 0)  # Green = ON. Helmet
                else:
                    color = (0, 0, 255)  # Red = NO Helmet
                    alert_triggered = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Record violation if needed
            if alert_triggered and not st.session_state.violation:
                st.warning("🚨 Violation Detected!")
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

                st.download_button("⬇️ Download Violation Snapshot", img_bytes, file_name=filename, mime="image/jpeg")

            # Show current frame
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.session_state.violation:
                        frame_placeholder.image(frame, channels="BGR", caption="Violation Detected", use_container_width=True)
                        break
                    else:
                        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            if not start_camera:
                break

    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        cap.release()

# ---------------------- Reset ----------------------
if reset_trigger:
    st.session_state.violation = False
    st.rerun()

# ---------------------- Violation Log ----------------------
st.markdown("---")
st.markdown("## 📋 Violation Log")

if st.session_state.history:
    log_data = [{"Timestamp": entry["timestamp"], "Class": entry["class"]} for entry in st.session_state.history]
    df = pd.DataFrame(log_data)
    df.index = df.index + 1
    st.dataframe(df)

    csv = df.to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Download Log as CSV", csv, "violation_log.csv", "text/csv")
else:
    st.info("✅ No helmet violations recorded yet.")
