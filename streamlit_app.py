import os
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
import zipfile

# Fix timezone issue for Streamlit Cloud
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

# ---------------------- App Config ----------------------
st.set_page_config(page_title="CapSure Live", page_icon="🪖", layout="wide")

# ---------------------- Load ONNX Model ----------------------
MODEL_ZIP = "best.zip"
MODEL_ONNX = "best.onnx"
LABELS = ["NO Helmet", "ON. Helmet"]
if not os.path.exists(MODEL_ONNX) and os.path.exists(MODEL_ZIP):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as z:
        z.extractall(".")

@st.cache_resource
def load_model():
    sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name

session, input_name = load_model()

# ---------------------- Preprocess / Postprocess ----------------------
def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def postprocess(outputs, thr):
    preds = outputs[0][0]
    results = []
    for p in preds:
        if len(p) >= 6:
            x1, y1, x2, y2 = (p[0], p[1], p[2], p[3])
            if len(p) == 6:
                conf, cls = p[4], int(p[5])
            else:
                obj_conf = p[4]
                cls_scores = p[5:]
                cls = int(np.argmax(cls_scores))
                conf = float(obj_conf * cls_scores[cls])
            if conf > thr:
                results.append((cls, conf, (int(x1), int(y1), int(x2), int(y2))))
    return results

# ---------------------- Video Transformer ----------------------
class HelmetDetector(VideoTransformerBase):
    def __init__(self, thr=0.3):
        self.thr = thr

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        inp = preprocess(img)
        outputs = session.run(None, {input_name: inp})
        detections = postprocess(outputs, self.thr)

        for cls_id, conf, (x1, y1, x2, y2) in detections:
            color = (0, 255, 0) if cls_id == 1 else (0, 0, 255)
            label = f"{LABELS[cls_id]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

# ---------------------- Sidebar ----------------------
st.sidebar.title("📷 Live Video Detection")
webrtc_ctx = st.sidebar.radio("Camera Stream Mode", ["On", "Off"]) == "On"
thr = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

# ---------------------- Main App ----------------------
st.title("🪖 CapSure Live Helmet Detection")
st.write("Uses live webcam feed for real-time helmet detection")

if webrtc_ctx:
    webrtc_streamer(
        key="helmet-detector",
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_transformer_factory=lambda: HelmetDetector(thr),
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
else:
    st.warning("🔴 Live stream is off. Switch On in sidebar to start detection.")
