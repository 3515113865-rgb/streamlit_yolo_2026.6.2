# Import All the Required Libraries
import cv2
import tempfile
import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# SAHI 导入（去掉旧版 yolov8 utils）
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Streamlit Application
st.set_page_config(page_title="YOLO11 Detector", page_icon="❤", layout="centered")
st.title("❤ YOLO11 - Image & Video + SAHI")
st.caption("Choose Task (Detection/Segmentation/Pose), then upload an Image or Video.")

# ---- Task Selection ----
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

# Load YOLO Model
@st.cache_resource(show_spinner=False)
def get_model(path: str):
    return YOLO(path)

model = get_model(model_path)

# SAHI Model（仅检测/分割可用，pose 不用）
@st.cache_resource(show_spinner=False)
def get_sahi_model(path: str, task_type: str):
    if task_type not in ["Object Detection", "Instance Segmentation"]:
        return None
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=path,
        confidence_threshold=0.25,
        device="cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() else "cpu"
    )

sahi_model = get_sahi_model(model_path, task)

# Mode Switch
mode = st.radio("Select Mode", ["Image", "Video (Live)"], horizontal=True)

# ---------------- IMAGE MODE ----------------
if mode == "Image":
    uploaded = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)

        with st.spinner("Running YOLO11 + SAHI..."):
            if sahi_model is not None:
                # 使用 SAHI 切片推理
                result_sahi = get_sliced_prediction(
                    img_np,
                    sahi_model,
                    slice_height=320,
                    slice_width=320,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2
                )
                annotated = result_sahi.image
            else:
                # Pose 直接用 YOLO
                result = model.predict(source=img_np, save=False, verbose=False)[0]
                annotated = result.plot(pil=True)

            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.subheader("Input")
                st.image(img, use_container_width=True)
            with col2:
                st.subheader("Detections (SAHI)")
                st.image(annotated, use_container_width=True)
    else:
        st.info("请先上传图片")

# ---------------- VIDEO MODE ----------------
else:
    uploaded = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "mkv"])
    conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    use_sahi = st.checkbox("Enable SAHI (better small objects, slower)", value=True if task != "Pose Estimation" else False)
    start = st.button("Start Detection")

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded.read())
            input_path = tmp_file.name
        st.caption(f"Input: {uploaded.name} | Task: {task} | Model: {model_path}")

    if start:
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        progress_bar = st.progress(0)

        if "input_path" not in locals():
            st.warning("请先上传视频！")
            st.stop()

        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        processed = 0

        with st.spinner(f"Running {task} on video..."):
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                if use_sahi and sahi_model is not None:
                    # SAHI 切片
                    result_sahi = get_sliced_prediction(
                        frame,
                        sahi_model,
                        slice_height=320,
                        slice_width=320,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2
                    )
                    annotated_bgr = result_sahi.image
                else:
                    # 原生 YOLO
                    results = model.predict(frame, conf=conf, verbose=False)[0]
                    annotated_bgr = results.plot()

                frame_placeholder.image(annotated_bgr, channels="BGR", use_container_width=True)
                processed += 1
                if total_frames:
                    progress_bar.progress(min(processed / total_frames, 1.0))
                info_placeholder.markdown(
                    f"**Frames:** {processed}/{total_frames if total_frames else '?'} · **Conf:** {conf} · **SAHI:** {'ON' if use_sahi and sahi_model else 'OFF'}"
                )
            cap.release()
        st.success("Finished Live Processing √")
        try:
            os.unlink(input_path)
        except Exception:
            pass