import streamlit as st
import numpy as np
from PIL import Image

# 🚨 CRITICAL: prevent cv2 import crash
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO

st.title("YOLO Object Detection (Cloud Safe Mode)")

@st.cache_resource
def load_model():
    # do NOT preload weights in unsafe mode
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

img_file = st.camera_input("Take a picture")

if img_file:
    image = Image.open(img_file)
    st.image(image, caption="Input Image")

    img = np.array(image)

    # 🚨 IMPORTANT: disable OpenCV pipeline
    results = model.predict(source=img, save=False, verbose=False, device='cpu')

    result_img = results[0].plot()

    st.image(result_img, caption="Detection Result")
