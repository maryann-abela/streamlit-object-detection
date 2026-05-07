import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("📷 Live Object Detection")
st.write("Take a photo using your webcam for object detection.")

# Open webcam
picture = st.camera_input("Take a picture")

if picture:
    image = Image.open(picture)

    st.image(image, caption="Captured Image")

    img_array = np.array(image)

    # Run YOLO detection
    results = model(img_array)

    # Draw detections
    annotated_frame = results[0].plot()

    st.image(
        annotated_frame,
        caption="Detected Objects",
        use_column_width=True
    )
