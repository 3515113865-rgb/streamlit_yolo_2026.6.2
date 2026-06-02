#Import All the Required Libraries
import os
os.environ['OPENCV_IO_ENABLE_OPENGL'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

#Streamlit Application
st.set_page_config(page_title="YOLO11 Detector", page_icon="❤", layout="centered")
st.title("❤ YOLO11 - Image & Video")
st.caption("Choose Task (Detection/ Segmentation/Pose Estimation), then upload an Image or Video. ")

#----Task Selection------
task = st.radio(
    "Select Task",
    ["Object Detection", "Instance Segmentation", "Pose Estimation"],
    horizontal=True
)

MODEL_MAP={
    "Object Detection": "yolo11n.pt",
    "Instance Segmentation": "yolo11n-seg.pt",
    "Pose Estimation": "yolo11n-pose.pt"
}

model_path = MODEL_MAP[task]

#Load YOLO Model
@st.cache_resource(show_spinner=False)
def get_model(path:str):
    return YOLO(path)

model = get_model(model_path)

#Mode
mode = st.radio("Select Mode", ["Image", "Video (Live)"], horizontal=True)

#IMAGE MODE
if mode == "Image":
    uploaded = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")

        with st.spinner("Running YOLO11..."):
            result = model.predict(source=np.array(img), save=True, verbose=False)[0]
            annotated_pil = result.plot(pil=True)
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.subheader("Input")
                st.image(img, use_container_width=True)
            with col2:
                st.subheader("Detections")
                st.image(annotated_pil, use_container_width=True)
    else:
        st.info("请先上传图片")

#VIDEO MODE (✅ 最终修复版)
else:
    uploaded = st.file_uploader("Upload a Video", type=["mp4"])
    conf = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
    start = st.button("Start Detection")

    input_path = None

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded.read())
            input_path = tmp_file.name

    if start:
        if not input_path:
            st.warning("请先上传视频")
            st.stop()

        frame_placeholder = st.empty()
        cap = cv2.VideoCapture(input_path)

        # 强制降低分辨率，云端必跑
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 推理
            res = model(frame, conf=conf, verbose=False)[0]
            annotated = res.plot()

            # 转 RGB + 显示
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, use_container_width=True)

        cap.release()
        st.success("✅ 视频处理完成！")