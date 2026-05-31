import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
page_title="Steel Surface Defect Detector",
page_icon="🔩",
layout="centered"
)

# Load TFLite model

interpreter = tf.lite.Interpreter(
model_path="model/cnn_steel_defect_model.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [
'Crazing',
'Inclusion',
'Patches',
'Pitted Surface',
'Rolled-in Scale',
'Scratches'
]

st.title("🔩 Steel Surface Defect Detector")
st.markdown(
"Upload a steel surface image and the model will identify the defect type."
)

uploaded = st.file_uploader(
"Upload Image",
type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded:

```
img = Image.open(uploaded).convert("RGB")
img = img.resize((224, 224))

st.image(img, caption="Uploaded Image", use_container_width=True)

arr = np.array(img, dtype=np.float32)

# Normalize if training used /255
arr = arr / 255.0

arr = np.expand_dims(arr, axis=0)

interpreter.set_tensor(
    input_details[0]['index'],
    arr
)

interpreter.invoke()

pred = interpreter.get_tensor(
    output_details[0]['index']
)

label = class_names[np.argmax(pred)]
confidence = float(np.max(pred) * 100)

st.success(
    f"Prediction: {label}"
)

st.info(
    f"Confidence: {confidence:.2f}%"
)
```
