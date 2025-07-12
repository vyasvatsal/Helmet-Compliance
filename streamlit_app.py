import os
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from playsound import playsound
import threading

# Config
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🪖", layout="wide")

# Constants
MODEL_PATH = "C:/Users/HP/Downloads/Intel Internship/Streamlit/capsure-env/best.onnx"
LABELS = ["NO Helmet", "ON. Helmet"]
ALARM_PATH = "C:/Users/HP/Downloads/Intel Internship/Streamlit/capsure-env/056857_alien-base-siren-40310.mp3"
SAVE_DIR = "C:/Users/HP/Downloads/Intel Internship/Streamlit/capsure-env/violations"
LOGO_PATH = "C:/Users/HP/Downloads/Intel Internship/Streamlit/capsure-env/CapSure logo.png"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load ONNX Model
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

session, input_name = load_model()

# Preprocess
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)  # HWC → CHW
    img_normalized = img_transposed.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

# Postprocess
def postprocess(outputs, threshold=0.3):  # Lowered threshold for sensitivity
    predictions = outputs[0][0]
    results = []
    for pred in predictions:
        if len(pred) < 6:
            continue
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf > threshold:
            results.append((int(cls), float(conf), (int(x1), int(y1), int(x2), int(y2))))
    return results

# Alarm
def play_alarm():
    threading.Thread(target=playsound, args=(ALARM_PATH,), daemon=True).start()

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

start_camera = st.sidebar.toggle("📷 Camera ON/OFF", value=False, key="cam_toggle")
reset_trigger = st.sidebar.button("🔁 RESET", use_container_width=True)

# Title
st.markdown("<h1 style='text-align:center; color:#3ABEFF;'>CapSure - Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# State init
if 'history' not in st.session_state:
    st.session_state.history = []

if 'violation' not in st.session_state:
    st.session_state.violation = False

# Placeholder for live frame display
frame_placeholder = st.empty()

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

                # Format timestamp
                dt = datetime.now(ZoneInfo("Asia/Kolkata"))
                formatted_time = dt.strftime("%I:%M:%S %p @ %d %B, %Y")
                filename = f"violation_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"

                # Encode frame to JPEG for download
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()

                # Add to session history (for log display only)
                st.session_state.history.insert(0, {
                    "timestamp": formatted_time,
                    "class": "NO Helmet",
                    "filename": filename,
                    "image_bytes": img_bytes
                })

                # Set violation flag
                st.session_state.violation = True
                st.warning("🚨 Violation Detected! Please RESET to continue.")

                # Auto download button
                st.download_button(
                    label=f"⬇️ Download Violation Snapshot",
                    data=img_bytes,
                    file_name=filename,
                    mime="image/jpeg"
                )

            # Display frame in center column
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.session_state.violation:
                        frame_placeholder.image(frame, channels="BGR", caption="Violation detected.", use_container_width=True)
                        break
                    else:
                        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            if not start_camera:
                break

    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        cap.release()

# Reset button logic
if reset_trigger:
    st.session_state.violation = False
    st.rerun()

# Defect Log Section
st.markdown("---")
st.markdown("## 📋 Defect Log (Recent Violations)")

if st.session_state.history:
    log_data = [
        {
            "Timestamp": entry["timestamp"],
            "Class": entry["class"]
        } for entry in st.session_state.history
    ]
    df = pd.DataFrame(log_data)
    df.index = df.index + 1
    st.dataframe(df)

    csv = df.to_csv(index=True).encode('utf-8')
    st.download_button("⬇️ Download Log as CSV", csv, "defect_log.csv", "text/csv")
else:
    st.info("No helmet violations recorded yet.")
