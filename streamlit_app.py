import os
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
import zipfile

# ──────────────────────────────
# CONFIGURATION
# ──────────────────────────────
st.set_page_config("CapSure - Helmet Detection", "🪖", layout="wide")

MODEL_ZIP = "best.zip"
MODEL_ONNX = "best.onnx"
LABELS = ["Helmet", "NO Helmet"]  # Flip this if needed!

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4

# ──────────────────────────────
# UNZIP MODEL IF NEEDED
# ──────────────────────────────
if not os.path.exists(MODEL_ONNX) and os.path.exists(MODEL_ZIP):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as z:
        z.extractall(".")

# ──────────────────────────────
# LOAD MODEL
# ──────────────────────────────
@st.cache_resource
def load_model():
    sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape
    return sess, name, shape

session, input_name, input_shape = load_model()
INPUT_SIZE = input_shape[2]

# ──────────────────────────────
# PREPROCESS
# ──────────────────────────────
def preprocess(img):
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ──────────────────────────────
# POSTPROCESS WITH NMS
# ──────────────────────────────
def nms(boxes, scores):
    indices = cv2.dnn.NMSBoxes(
        [list(map(int, b)) for b in boxes],
        scores, CONF_THRESHOLD, IOU_THRESHOLD
    )
    return indices.flatten() if len(indices) > 0 else []

def postprocess(outs):
    preds = outs[0][0]
    boxes, scores, classes = [], [], []

    for pred in preds:
        x_c, y_c, w, h = pred[:4]
        obj_conf = pred[4]
        cls_scores = pred[5:]
        cls_id = np.argmax(cls_scores)
        cls_conf = cls_scores[cls_id]
        score = obj_conf * cls_conf

        if score > CONF_THRESHOLD:
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            classes.append(int(cls_id))

    idxs = nms(boxes, scores)
    return [(classes[i], scores[i], boxes[i]) for i in idxs]

# ──────────────────────────────
# Streamlit Interface
# ──────────────────────────────
st.title("🪖 Live Helmet Detection")
start_cam = st.sidebar.toggle("📷 Camera ON", value=False)
frame_placeholder = st.empty()
warn_placeholder = st.empty()

if start_cam:
    cap = cv2.VideoCapture(0)
    st.sidebar.success("🔴 Live Camera Started")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Camera frame not available.")
                break

            inp = preprocess(frame)
            outs = session.run(None, {input_name: inp})
            results = postprocess(outs)

            alert = False
            for clsid, conf, (x1, y1, x2, y2) in results:
                label = LABELS[clsid] if clsid < len(LABELS) else f"Class {clsid}"
                color = (0, 255, 0) if "Helmet" in label else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if "NO Helmet" in label:
                    alert = True

            # Show warning if violation
            if alert:
                warn_placeholder.warning("🚨 NO Helmet Detected!")
            else:
                warn_placeholder.empty()

            frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        cap.release()
        st.sidebar.info("🟡 Camera stopped.")
else:
    st.sidebar.info("Toggle to start camera")
