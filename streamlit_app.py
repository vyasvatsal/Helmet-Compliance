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
import tempfile
import zipfile

# Config
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🪖", layout="wide")

# Constants
MODEL_PATH = "best.onnx"  # Updated to use relative path
MODEL_ZIP = "best.zip"    # Zip file path
LABELS = ["NO Helmet", "ON. Helmet"]
ALARM_PATH = "alarm.mp3"  # Updated to use relative path
SAVE_DIR = "violations"
LOGO_PATH = "logo.png"  # Updated to use relative path
os.makedirs(SAVE_DIR, exist_ok=True)

# Load ONNX Model
@st.cache_resource
def load_model():
    try:
        # First try to load the extracted model
        if os.path.exists(MODEL_PATH):
            session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
            return session, session.get_inputs()[0].name
        
        # If not found, try to extract from zip
        elif os.path.exists(MODEL_ZIP):
            with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
                zip_ref.extractall('.')
            if os.path.exists(MODEL_PATH):
                session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
                return session, session.get_inputs()[0].name
        
        st.error(f"Model file not found. Please ensure {MODEL_PATH} or {MODEL_ZIP} exists in the current directory.")
        return None, None
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

session, input_name = load_model()

# Preprocess
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)  # HWC → CHW
    img_normalized = img_transposed.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

# Postprocess
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

# Alarm
def play_alarm():
    try:
        if os.path.exists(ALARM_PATH):
            threading.Thread(target=playsound, args=(ALARM_PATH,), daemon=True).start()
        else:
            st.warning("Alarm file not found. Please ensure alarm.mp3 exists.")
    except Exception as e:
        st.error(f"Error playing alarm: {e}")

# Sidebar UI
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)

st.sidebar.markdown(
    """
    <h1 style='text-align:center; color:yellow; font-size: 36px;'>CapSure</h1>
    <h2 style='text-align:center; color:yellow; font-size: 20px;'>Real-time Helmet Compliance Detection</h2>
    """, 
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Camera toggle in sidebar
camera_enabled = st.sidebar.toggle("📷 Enable Camera", value=False, key="camera_toggle")
reset_trigger = st.sidebar.button("🔁 RESET", use_container_width=True)

# Title
st.markdown("<h1 style='text-align:center; color:#3ABEFF;'>CapSure - Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# State initialization
if 'history' not in st.session_state:
    st.session_state.history = []

if 'violation' not in st.session_state:
    st.session_state.violation = False

if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if camera_enabled:
        st.info("🎥 Camera is enabled. Please allow webcam access when prompted.")
        
        # Streamlit webcam component
        camera_photo = st.camera_input("Take a photo", key="webcam")
        
        if camera_photo is not None:
            # Convert the photo to OpenCV format
            bytes_data = camera_photo.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None and session is not None:
                # Process the frame
                img_input = preprocess(frame)
                outputs = session.run(None, {input_name: img_input})
                detections = postprocess(outputs)
                
                alert_triggered = False
                
                # Draw detections on frame
                for cls_id, conf, (x1, y1, x2, y2) in detections:
                    label = LABELS[cls_id]
                    color = (0, 255, 0) if label == "ON. Helmet" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    if label == "NO Helmet":
                        alert_triggered = True
                
                # Display processed frame
                st.image(frame, channels="BGR", caption="Processed Frame", use_container_width=True)
                
                # Handle violation detection
                if alert_triggered and not st.session_state.violation:
                    play_alarm()
                    
                    # Format timestamp
                    dt = datetime.now(ZoneInfo("Asia/Kolkata"))
                    formatted_time = dt.strftime("%I:%M:%S %p @ %d %B, %Y")
                    filename = f"violation_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
                    
                    # Encode frame to JPEG for download
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    img_bytes = img_encoded.tobytes()
                    
                    # Add to session history
                    st.session_state.history.insert(0, {
                        "timestamp": formatted_time,
                        "class": "NO Helmet",
                        "filename": filename,
                        "image_bytes": img_bytes
                    })
                    
                    # Set violation flag
                    st.session_state.violation = True
                    st.warning("🚨 Violation Detected! Please RESET to continue.")
                    
                    # Download button for violation snapshot
                    st.download_button(
                        label=f"⬇️ Download Violation Snapshot",
                        data=img_bytes,
                        file_name=filename,
                        mime="image/jpeg"
                    )
                
                # Display detection results
                if detections:
                    st.markdown("### Detection Results:")
                    for cls_id, conf, (x1, y1, x2, y2) in detections:
                        label = LABELS[cls_id]
                        status_emoji = "✅" if label == "ON. Helmet" else "❌"
                        st.markdown(f"{status_emoji} **{label}** (Confidence: {conf:.2f})")
                else:
                    st.info("No helmets detected in the frame.")
            elif session is None:
                st.error("Model not loaded. Please check if the model file exists.")
    else:
        st.info("📷 Click the toggle button in the sidebar to enable camera access.")

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

# Instructions
st.markdown("---")
st.markdown("### 📋 Instructions:")
st.markdown("""
1. **Enable Camera**: Use the toggle button in the sidebar to enable webcam access
2. **Take Photo**: Click the camera button to capture an image for analysis
3. **View Results**: Detection results will be displayed below the image
4. **Reset**: Use the RESET button if an alarm is triggered
5. **Download**: Violation snapshots and logs can be downloaded automatically
""")
