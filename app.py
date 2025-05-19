import streamlit as st
import numpy as np
import cv2
import gdown
from tensorflow.keras.models import load_model
from PIL import Image


model = load_model('Intel_image_classification.h5')

# url = "https://drive.google.com/uc?id=https://drive.google.com/drive/folders/1on6kyGJl40OnKAOhFZsk7A3_8prkZx_r?usp=sharing"
# output = "Intel_image_classification.h5"
# gdown.download(url, output, quiet=False, fuzzy=True)

# model = load_model(output)

class_labels = {
    0: 'Buildings',
    1: 'Forest',
    2: 'Glacier',
    3: 'Mountain',
    4: 'Sea',
    5: 'Street'
}

st.title("Intel Image Classification")
st.write("Upload an image to classify it into one of six categories: Buildings, Forest, Glacier, Mountain, Sea, or Street.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)
    
    # Preprocess image
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  
    img = img.reshape(1, 150, 150, 3)   
    
    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels.get(predicted_class, "Unknown")
    
    # Display result
    st.markdown(f"### Predicted Class: **{predicted_label}**")
