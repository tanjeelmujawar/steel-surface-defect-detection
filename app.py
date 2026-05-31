import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("cnn_steel_defect_model.h5")
class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted Surface', 'Rolled-in Scale', 'Scratches']

st.title("🔩 Steel Surface Defect Detector")
uploaded = st.file_uploader("Upload a steel surface image", type=["jpg","png","bmp"])

if uploaded:
    img = Image.open(uploaded).resize((224, 224))
    st.image(img, caption="Uploaded Image")
    arr = np.expand_dims(np.array(img), axis=0)
    pred = model.predict(arr)
    label = class_names[np.argmax(pred)]
    conf = np.max(pred) * 100
    st.success(f"Prediction: **{label}** ({conf:.1f}% confidence)")