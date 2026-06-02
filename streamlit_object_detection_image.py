#Import All the Required Libraries
import os
os.environ['OPENCV_IO_ENABLE_OPENGL'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
import cv2

import streamlit as st
from networkx.drawing import layout
from ultralytics import YOLO
from PIL import Image
import numpy as np

#Streamlit Application

st.set_page_config(page_title="YOL011 Image Detector",page_icon="❤",layout="centered")
st.title("❤ YOLO11 - Image Object Detection (Super Simple)")
st.caption("Upload an Image. I'll run YOLO11 and show the result")

#Load the YOLO11 nano model once and cache it
@st.cache_resource(show_spinner=False)
def get_model():
    return YOLO("yolo11n.pt")

model = get_model()

uploaded = st.file_uploader("Upload an Image (PNG/JPG/WebP)",type=["png","jpg","webp"])

if uploaded is None:
    st.info("⭐Upload an Image to Begin.")
    st.stop()

# Read Input
img = Image.open(uploaded).convert("RGB")

#Run Detection
with st.spinner("Running YOLO11..."):
    result = model.predict(source=np.array(img), save=True)[0]

try:
    annotated_pil = result.plot(pil=True)  # PIL.Image(RGB)
except TypeError:
    annotated_bgr = result.plot()
    annotated_rgb = np.ascontiguousarray(annotated_bgr[:, :, ::-1])  # BGR→RGB
    annotated_pil = Image.fromarray(annotated_rgb)

#Two Columns; Left = Input, Right = Output
col1, col2 = st.columns(2, gap = "medium")
with col1:
    st.subheader("Input")
    st.image(img, use_container_width=True)
with col2:
    st.subheader("Detections")
    st.image(annotated_pil, use_container_width = True)