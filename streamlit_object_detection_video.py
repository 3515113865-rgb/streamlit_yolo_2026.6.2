import os
os.environ['OPENCV_IO_ENABLE_OPENGL'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="YOLO11 Detector",page_icon="❤",layout="centered")
st.title("❤ YOLO11 - Image & Video")
st.caption("Choose Task,then upload an Image or Video. ")

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

@st.cache_resource(show_spinner=False)
def get_model(path:str):
    return YOLO(path)
model = get_model(model_path)

mode = st.radio("Select Mode", ["Image", "Video"], horizontal=True)

# 图片模块不变
if mode == "Image":
    uploaded = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        with st.spinner("Running YOLO11..."):
            result = model.predict(source=np.array(img), save=False, verbose=False)[0]
            annotated_pil = result.plot(pil=True)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input")
                st.image(img, use_container_width=True)
            with col2:
                st.subheader("Detections")
                st.image(annotated_pil, use_container_width=True)

# 【全部重写视频代码，核心三处改动：stream=True、分辨率320、清空opencv缓存】
else:
    uploaded = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    conf = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
    start = st.button("Start Detection")
    temp_path = None

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(uploaded.read())
            temp_path = f.name

    if start and temp_path:
        frame_box = st.empty()
        cap = cv2.VideoCapture(temp_path)
        # 缩小画面减少算力
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1) # 关键：opencv只存1帧缓存，杜绝堆帧卡死

        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            # 重点：加stream=True流式推理，解决一次性加载全帧卡死
            for res in model(frame,conf=conf,verbose=False,imgsz=320,stream=True):
                out = res.plot()
                frame_box.image(out,channels="BGR",width=520)
            time.sleep(0.032) # 固定30fps

        cap.release()
        os.remove(temp_path)
        st.success("处理完毕")
    elif start and not temp_path:
        st.warning("先上传视频！")