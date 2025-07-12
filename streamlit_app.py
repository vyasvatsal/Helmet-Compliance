import os
import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
import pandas as pd
import base64
import zipfile
from PIL import Image
import io

# Config
st.set_page_config(page_title="CapSure - Helmet Detection", page_icon="🪖", layout="wide")

# Constants
MODEL_PATH = "best.onnx"
MODEL_ZIP_PATH = "best.zip"
LOGO_PATH = "logo.png"
LABELS = ["NO Helmet", "ON. Helmet"]


# Unzip model if not already extracted
if not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")

# Check if running locally or on cloud
def is_local():
    return os.path.exists('/dev/video0') or os.path.exists('/dev/video1')

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file 'best.onnx' not found. Please ensure the model is properly extracted.")
        st.stop()
    
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        return session, session.get_inputs()[0].name
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Only load model if it exists
try:
    session, input_name = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.warning("⚠️ Model not found. Upload detection will work, but real-time detection is disabled.")

# Preprocess image
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)  # HWC → CHW
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

# Process uploaded image
def process_uploaded_image(image):
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv = img_array
    
    # Run inference
    img_input = preprocess(img_cv)
    outputs = session.run(None, {input_name: img_input})
    detections = postprocess(outputs)
    
    # Draw bounding boxes
    for cls_id, conf, (x1, y1, x2, y2) in detections:
        label = LABELS[cls_id]
        color = (0, 255, 0) if label == "ON. Helmet" else (0, 0, 255)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_cv, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Convert back to PIL for display
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), detections

# Sidebar UI
st.sidebar.image(LOGO_PATH, use_container_width=True)

st.sidebar.markdown(
    """
    <h1 style='text-align:center; color:yellow; font-size: 36px;'>CapSure</h1>
    <h2 style='text-align:center; color:yellow; font-size: 20px;'>Helmet Compliance Detection</h2>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Detection mode selection
detection_mode = st.sidebar.selectbox(
    "🔍 Detection Mode",
    ["📷 Live Camera (Local Only)", "📁 Upload Image", "🎥 Upload Video"]
)

# Title
st.markdown("<h1 style='text-align:center; color:#3ABEFF;'>CapSure - Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Init session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'violation' not in st.session_state:
    st.session_state.violation = False

# Live Camera Mode (Local Only)
if detection_mode == "📷 Live Camera (Local Only)":
    if not is_local():
        st.error("🚫 Live camera is only available when running locally.")
        st.info("💡 **To use live camera:**")
        st.code("streamlit run app.py")
        st.info("📁 **For Streamlit Cloud, use 'Upload Image' or 'Upload Video' mode**")
    else:
        start_camera = st.sidebar.toggle("📷 Camera ON/OFF", value=False)
        reset_trigger = st.sidebar.button("🔁 RESET", use_container_width=True)
        
        if start_camera and model_loaded:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Camera not accessible. Please check camera permissions.")
            else:
                st.success("🎥 Live camera active")
                frame_placeholder = st.empty()
                
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        img_input = preprocess(frame)
                        outputs = session.run(None, {input_name: img_input})
                        detections = postprocess(outputs)
                        
                        for cls_id, conf, (x1, y1, x2, y2) in detections:
                            label = LABELS[cls_id]
                            color = (0, 255, 0) if label == "ON. Helmet" else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                        
                        if not start_camera:
                            break
                            
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    cap.release()

# Upload Image Mode
elif detection_mode == "📁 Upload Image":
    st.markdown("### 📁 Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to detect helmet compliance"
    )
    
    if uploaded_file is not None and model_loaded:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### Detection Results")
            with st.spinner("Processing..."):
                result_image, detections = process_uploaded_image(image)
                st.image(result_image, use_container_width=True)
        
        # Show detection results
        if detections:
            st.markdown("### 🎯 Detection Summary")
            for i, (cls_id, conf, bbox) in enumerate(detections):
                label = LABELS[cls_id]
                status = "✅ Compliant" if label == "ON. Helmet" else "❌ Violation"
                st.write(f"**Detection {i+1}:** {status} - {label} ({conf:.2f})")
                
                # Log violations
                if label == "NO Helmet":
                    now = datetime.now()
                    formatted_time = now.strftime("%I:%M:%S %p @ %d %B, %Y")
                    
                    # Convert result image to bytes for download
                    img_bytes = io.BytesIO()
                    result_image.save(img_bytes, format='JPEG')
                    img_bytes = img_bytes.getvalue()
                    
                    st.session_state.history.insert(0, {
                        "timestamp": formatted_time,
                        "class": "NO Helmet",
                        "filename": f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg",
                        "image_bytes": img_bytes
                    })
                    
                    st.download_button(
                        label="⬇️ Download Violation Image",
                        data=img_bytes,
                        file_name=f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg",
                        mime="image/jpeg"
                    )
        else:
            st.info("No helmet detections found in the image.")

# Upload Video Mode
elif detection_mode == "🎥 Upload Video":
    st.markdown("### 🎥 Upload Video for Detection")
    st.info("⚠️ Video processing may take time depending on video length and size.")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video to detect helmet compliance"
    )
    
    if uploaded_video is not None and model_loaded:
        # Save uploaded video temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        if st.button("🔄 Process Video"):
            cap = cv2.VideoCapture("temp_video.mp4")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            progress_bar = st.progress(0)
            violations_found = 0
            
            for i in range(0, frame_count, 30):  # Process every 30th frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                img_input = preprocess(frame)
                outputs = session.run(None, {input_name: img_input})
                detections = postprocess(outputs)
                
                for cls_id, conf, bbox in detections:
                    if LABELS[cls_id] == "NO Helmet":
                        violations_found += 1
                        break
                
                progress_bar.progress(i / frame_count)
            
            cap.release()
            os.remove("temp_video.mp4")
            
            st.success(f"✅ Video processed! Found {violations_found} potential violations.")

# Defect Log
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

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <p>💡 <strong>Tip:</strong> For best results, ensure good lighting and clear visibility of persons in images/videos.</p>
    <p>🔧 <strong>Local Mode:</strong> Run <code>streamlit run app.py</code> for live camera access.</p>
    </div>
    """,
    unsafe_allow_html=True
)
