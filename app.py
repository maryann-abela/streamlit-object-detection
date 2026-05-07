import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    # Auto-download official YOLO model
    return YOLO("yolov8n.pt")

model = load_model()

st.title("📷 Object Detection App")

picture = st.camera_input("Take a picture")

if picture:
    image = Image.open(picture)

    st.image(image, caption="Captured Image")

    img_array = np.array(image)

    results = model(img_array)

    annotated_frame = results[0].plot()

    st.image(
        annotated_frame,
        caption="Detected Objects",
        use_column_width=True
    )
