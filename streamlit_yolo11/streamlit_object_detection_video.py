#Import All the Required Libraries
import cv2
import tempfile
import os
import streamlit as st
from networkx.drawing import layout
from ultralytics import YOLO
from PIL import Image
import numpy as np

#Streamlit Application

st.set_page_config(page_title="YOL011 Detector",page_icon="❤",layout="centered")
st.title("❤ YOLO11 - Image & Video")
st.caption("Choose Task (Detection/ Segmentation/Pose Estimation),then upload an Image or Video. ")

#----Task Selection:Choose the model family------
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

#Load YOL0 Model once per chosen task
@st.cache_resource(show_spinner=False)
def get_model(path:str):
    return YOLO(path)

model = get_model(model_path)

#ModeSwitch
mode = st.radio("Select Mode", ["Image", "Video (Live)"], horizontal=True)

#IMAGE MODE
if mode == "Image":
    uploaded = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "webp"])
    # Read Input
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")

        # Run Detection
        with st.spinner("Running YoL011..."):
            result = model.predict(source=np.array(img), save=True, verbose=True)[0]
            annotated_pil = result.plot(pil=True)  # Works for Detection / Segmentation / Pose Estimation
            # Two Columns; Left = Input, Right = Output
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.subheader("Input")
                st.image(img, use_container_width=True)
            with col2:
                st.subheader("Detections")
                st.image(annotated_pil, use_container_width=True)
    else:
        st.info("请先上传图片")


#VIDEO MODE(LIVE)
else:
    uploaded = st.file_uploader("Upload an Video", type=["mp4", "mov", "avi", "mkv"])
    conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    start = st.button("start Detection")


    # Save the Uploaded Video to a tempfile so OpencV can readit
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded.read())
            input_path = tmp_file.name

        st.caption(f"Input File: {uploaded.name} + Task：{task} + Model:{model_path}")

    if start:
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        progress_bar = st.progress(0)

        if 'input_path' not in locals():
            st.warning("请先上传视频！")
            st.stop()

        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        processed = 0

        with st.spinner(f"Running {task} on video (live)...."):
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                # Run the selected task on every frame
                results = model.predict(frame, conf=conf, verbose=True)[0]
                annotated_bgr = results.plot()

                # Show Live Frame
                frame_placeholder.image(annotated_bgr, channels="BGR", use_container_width=True)

                processed += 1

                # Update progress bar
                if total_frames:
                    progress_bar.progress(min(processed / total_frames, 1.0))

                # Update Info
                info_placeholder.markdown(
                    f"**Frames processed:** {processed} / {total_frames if total_frames else 'unknown'}"
                    f" · **Confidence:** {conf}"
                )
            cap.release()
        st.success("Finished Live Processing √ ")
        try:
            os.unlink(input_path)
        except Exception:
            pass



