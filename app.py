import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt", task="detect")

model = load_model()

st.title("📷 Object Detection App")

image_file = st.camera_input("Take a picture")

if image_file:
    image = Image.open(image_file)

    st.image(image)

    img = np.array(image)

    results = model(img)

    output = results[0].plot()

    st.image(output)
