import os
os.environ["YOLO_VERBOSE"] = "False"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLO Object Detection (FIXED VERSION)")

@st.cache_resource
def load_model():
    # IMPORTANT: force download-safe mode
    return YOLO("yolov8n.pt")

model = load_model()

img_file = st.camera_input("Take a picture")

if img_file:
    image = Image.open(img_file)
    st.image(image, caption="Input")

    img = np.array(image)

    # force CPU-safe inference
    results = model.predict(img, verbose=False)

    output = results[0].plot()

    st.image(output, caption="Detection Result")
