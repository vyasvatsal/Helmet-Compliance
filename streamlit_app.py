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

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🪖", layout="wide")

MODEL_ZIP = "best.zip"
MODEL_ONNX = "best.onnx"
LABELS = ["NO Helmet", "ON. Helmet"]  # Check this against your model's class index!
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4  # for NMS

# ─────────────────────────────────────────────────────────
# UNZIP MODEL IF NEEDED
# ─────────────────────────────────────────────────────────
if not os.path.exists(MODEL_ONNX) and os.path.exists(MODEL_ZIP):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as z:
        z.extractall(".")

# ─────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    return sess, input_name, input_shape

session, input_name, input_shape = load_model()
INPUT_SIZE = input_shape[2]  # Typically 640 for YOLOv5

# ─────────────────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────────────────
def preprocess(image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel RGB image.")
    img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ─────────────────────────────────────────────────────────
# POSTPROCESS + NMS
# ─────────────────────────────────────────────────────────
def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(
        bboxes=[list(map(int, b)) for b in boxes],
        scores=scores,
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=iou_threshold
    )
    return indices.flatten() if len(indices) > 0 else []

def postprocess(outputs):
    predictions = outputs[0][0]
    boxes, scores, classes = [], [], []

    for pred in predictions:
        x_c, y_c, w, h = pred[0:4]
        obj_conf = pred[4]
        cls_scores = pred[5:]
        cls_id = np.argmax(cls_scores)
        cls_conf = cls_scores[cls_id]
        score = obj_conf * cls_conf

        if score > CONFIDENCE_THRESHOLD:
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            classes.append(int(cls_id))

    idxs = non_max_suppression(boxes, scores, IOU_THRESHOLD)
    results = [(classes[i], scores[i], boxes[i]) for i in idxs]
    return results

# ─────────────────────────────────────────────────────────
# STATE SETUP
# ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "violated" not in st.session_state:
    st.session_state.violated = False

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.markdown(
    "<h1 style='text-align:center; color:yellow;'>CapSure</h1><h2 style='text-align:center; color:yellow;'>Helmet Detection</h2>",
    unsafe_allow_html=True
)
start = st.sidebar.toggle("📷 Camera ON/OFF")
if st.sidebar.button("🔁 RESET"):
    st.session_state.violated = False

# ─────────────────────────────────────────────────────────
# MAIN TITLE
# ─────────────────────────────────────────────────────────
st.title("🪖 Helmet Compliance Detection")

if start and not st.session_state.violated:
    img_file = st.camera_input("📸 Capture Image")
    if img_file:
        img_pil = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        try:
            inp = preprocess(frame)
            outs = session.run(None, {input_name: inp})
            detections = postprocess(outs)

            alert = False
            for clsid, conf, (x1, y1, x2, y2) in detections:
                label = LABELS[clsid] if clsid < len(LABELS) else f"Class {clsid}"
                color = (0, 255, 0) if label == "ON. Helmet" else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if label == "NO Helmet":
                    alert = True

            st.image(frame, channels="BGR", use_column_width=True)

            if alert:
                st.warning("🚨 Helmet Violation Detected!")
                now = datetime.now(ZoneInfo("Asia/Kolkata"))
                ts = now.strftime("%I:%M:%S %p @ %d %B, %Y")
                fn = f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                _, buffer = cv2.imencode(".jpg", frame)

                st.session_state.history.insert(0, {
                    "ts": ts,
                    "class": "NO Helmet",
                    "bytes": buffer.tobytes(),
                    "fn": fn
                })
                st.session_state.violated = True

                st.download_button("⬇️ Download Snapshot", buffer.tobytes(), file_name=fn, mime="image/jpeg")
        except Exception as e:
            st.error(f"❌ Error in processing: {e}")
    elif st.session_state.violated:
        st.warning("⚠️ Detection paused due to previous violation. Click RESET to continue.")

# ─────────────────────────────────────────────────────────
# VIOLATION LOG
# ─────────────────────────────────────────────────────────
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
