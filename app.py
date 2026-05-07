import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load model
model = YOLO("yolov8n.pt")

st.title("Object Detection App (YOLO)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image for YOLO
    img_array = np.array(image)

    # Run detection
    results = model(img_array)

    # Plot results
    annotated_img = results[0].plot()

    st.image(annotated_img, caption="Detected Objects", use_column_width=True)
