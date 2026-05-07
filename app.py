import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLO Object Detection")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)

    results = model(img_array)

    annotated_frame = results[0].plot()

    st.image(
        annotated_frame,
        caption="Detected Objects"
    )
