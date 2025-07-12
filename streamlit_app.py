import os
import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
import zipfile

# For Streamlit Cloud (timezone fix)
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

# ---------------------- App Config ----------------------
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🪖", layout="wide")

# ---------------------- Session State ----------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "violated" not in st.session_state:
    st.session_state.violated = False

# ---------------------- Constants ----------------------
MODEL_ZIP = "best.zip"
MODEL_ONNX = "best.onnx"
LABELS = ["NO Helmet", "ON. Helmet"]
LOGO_PATH = "logo.png"

# ---------------------- Extract ONNX ----------------------
if not os.path.exists(MODEL_ONNX) and os.path.exists(MODEL_ZIP):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as z:
        z.extractall(".")

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

# ---------------------- Universal Postprocess ----------------------
def postprocess(outputs, conf_threshold=0.3):
    predictions = outputs[0][0]

    results = []
    for pred in predictions:
        if len(pred) == 6:
            # [x1, y1, x2, y2, conf, class_id]
            x1, y1, x2, y2, conf, cls_id = pred
            if conf > conf_threshold:
                results.append((int(cls_id), float(conf), (int(x1), int(y1), int(x2), int(y2))))
        elif len(pred) >= 6:
            # YOLOv5-style: [x_center, y_center, w, h, obj_conf, cls_scores...]
            x_center, y_center, width, height = pred[:4]
            objectness = pred[4]
            class_scores = pred[5:]
            cls_id = np.argmax(class_scores)
            cls_conf = class_scores[cls_id]
            confidence = objectness * cls_conf
            if confidence > conf_threshold:
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                results.append((cls_id, float(confidence), (x1, y1, x2, y2)))
    return results

# ---------------------- Sidebar UI ----------------------
st.sidebar.image(LOGO_PATH, use_column_width=True)
st.sidebar.markdown(
    "<h1 style='text-align:center; color:yellow;'>CapSure</h1><h2 style='text-align:center; color:yellow;'>Helmet Detection</h2>",
    unsafe_allow_html=True
)
start = st.sidebar.toggle("📷 Camera ON/OFF")
if st.sidebar.button("🔁 RESET"):
    st.session_state.violated = False

# ---------------------- Main Title ----------------------
st.markdown("<h1 style='text-align:center; color:#3ABEFF;'>🪖 CapSure - Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------- Detection Workflow ----------------------
if start and not st.session_state.violated:
    img_file = st.camera_input("📸 Capture an image for helmet detection")

    if img_file:
        # Convert to OpenCV
        img_pil = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Inference
        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})
        detections = postprocess(outputs)

        # Draw boxes
        alert = False
        for cls_id, conf, (x1, y1, x2, y2) in detections:
            label = LABELS[cls_id] if cls_id < len(LABELS) else f"Class {cls_id}"
            color = (0, 255, 0) if cls_id == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if cls_id == 0:
                alert = True

        # Show annotated image
        st.image(frame, channels="BGR", use_column_width=True)

        # If violation detected
        if alert:
            st.warning("🚨 Helmet Violation Detected!")
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            timestamp = now.strftime("%I:%M:%S %p @ %d %B, %Y")
            filename = f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            _, buffer = cv2.imencode(".jpg", frame)

            # Save to session history
            st.session_state.history.insert(0, {
                "ts": timestamp,
                "class": "NO Helmet",
                "bytes": buffer.tobytes(),
                "fn": filename
            })
            st.session_state.violated = True

            st.download_button("⬇️ Download Violation Snapshot", buffer.tobytes(), file_name=filename, mime="image/jpeg")

    elif st.session_state.violated:
        st.warning("⚠️ Detection paused due to previous violation. Click RESET to continue.")

# ---------------------- Violation Log ----------------------
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
