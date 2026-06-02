import os

os.environ['OPENCV_IO_ENABLE_OPENGL'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ==================== 页面配置 ====================
st.set_page_config(page_title="YOLO11 Detector", page_icon="❤", layout="centered")
st.title("❤ YOLO11 - 图片 & 视频检测")

# ==================== 模型选择 ====================
task = st.radio(
    "Select Task",
    ["Object Detection", "Instance Segmentation", "Pose Estimation"],
    horizontal=True
)

MODEL_MAP = {
    "Object Detection": "yolo11n.pt",
    "Instance Segmentation": "yolo11n-seg.pt",
    "Pose Estimation": "yolo11n-pose.pt"
}
model_path = MODEL_MAP[task]


@st.cache_resource
def get_model(path):
    return YOLO(path)


model = get_model(model_path)

# ==================== 模式选择 ====================
mode = st.radio("Select Mode", ["Image", "Video"], horizontal=True)

# ==================== 图片模式 ====================
if mode == "Image":
    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        with st.spinner("Detecting..."):
            result = model.predict(np.array(img), verbose=False)[0]
            res_img = result.plot(pil=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(img)
        with col2:
            st.subheader("Output")
            st.image(res_img)

# ==================== 【最终稳定版：视频处理 + 生成MP4 + 播放】 ====================
else:
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    conf = st.slider("Confidence", 0.1, 0.9, 0.25)

    if uploaded_video is not None:
        # 保存临时文件
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_in.write(uploaded_video.read())
        temp_in.close()

        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        st.success("上传成功 → 点击开始处理")

        if st.button("Start Video Detection"):
            cap = cv2.VideoCapture(temp_in.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))

            progress = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current = 0

            with st.spinner("Processing video..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # YOLO 检测
                    res = model.predict(frame, conf=conf, verbose=False)[0]
                    frame = res.plot()

                    out.write(frame)
                    current += 1
                    progress.progress(min(current / frame_count, 1.0))

            cap.release()
            out.release()

            st.success("✅ 视频处理完成！")

            # 直接播放处理好的视频（必动、必流畅）
            st.video(temp_out.name)

            os.unlink(temp_in.name)
            os.unlink(temp_out.name)