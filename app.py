import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("📷 YOLO Object Detection (Stable Version)")

image_file = st.camera_input("Take a picture")

if image_file:
    image = Image.open(image_file)

    st.image(image, caption="Input Image")

    img_array = np.array(image)

    results = model(img_array)

    output_image = results[0].plot()

    st.image(output_image, caption="Detection Result")
