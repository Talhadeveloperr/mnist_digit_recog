# streamlit_app/app.py

import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import os

# Set page config
st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

# Load model and scaler
MODEL_PATH = os.path.join("..", "model", "best_model.pkl")
SCALER_PATH = os.path.join("..", "model", "scaler.pkl")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

st.title("🧠 MNIST Digit Recognition")
st.write("Upload a handwritten digit image or try a sample to see the model's prediction.")

# Sample image for demo (use an actual digit image from sklearn.datasets)
sample_digit = np.array([
    [0.00, 0.00, 0.00, 5.31, 6.71, 0.00, 0.00, 0.00],
    [0.00, 0.00, 3.36, 7.62, 7.62, 2.14, 0.00, 0.00],
    [0.00, 0.00, 6.71, 7.62, 7.62, 6.41, 0.00, 0.00],
    [0.00, 0.00, 1.83, 1.68, 5.18, 7.62, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 1.22, 7.62, 0.00, 0.00],
    [0.00, 0.00, 2.67, 4.58, 6.86, 7.62, 7.01, 0.15],
    [0.00, 0.00, 6.71, 7.62, 7.62, 7.62, 7.62, 3.36],
    [0.00, 0.00, 0.00, 0.00, 0.00, 2.14, 6.41, 6.71]
])  # a sample image (looks like digit 8)

# Show sample button
if st.button("🎯 Try Sample Image"):
    st.image(Image.fromarray((sample_digit * 16).astype(np.uint8)).resize((200, 200)), caption="Sample Digit")
    flat = sample_digit.flatten().reshape(1, -1)
    scaled = scaler.transform(flat)
    prediction = model.predict(scaled)[0]
    st.success(f"Predicted Digit: {prediction}")

# Divider
st.markdown("---")

# Upload file
uploaded_file = st.file_uploader("📤 Upload a 28x28 handwritten digit image (png/jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('L')
    img = ImageOps.invert(img)  # invert for white background
    img_resized = img.resize((8, 8))
    
    # Display image
    st.image(img_resized.resize((200, 200)), caption="Uploaded Digit")

    # Prepare for prediction
    img_np = np.array(img_resized).astype('float64')
    img_np = (img_np / 255.0) * 16  # scale to 0-16 as in sklearn dataset
    flat = img_np.flatten().reshape(1, -1)
    scaled = scaler.transform(flat)

    # Predict
    prediction = model.predict(scaled)[0]
    st.success(f"Predicted Digit: {prediction}")
