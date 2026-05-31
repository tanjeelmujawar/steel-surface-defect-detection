import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import cv2



# ------------------------------------------
# STEP 1 : Dataset Path
# ------------------------------------------
data_dir = r"D:\CP2\NEU-CLS"

# ------------------------------------------
# STEP 2 : Load Dataset
# ------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224,224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224,224),
    batch_size=32
)

class_names = [
    'Crazing',
    'Inclusion',
    'Patches',
    'Pitted Surface',
    'Rolled-in Scale',
    'Scratches'
]

num_classes = 6




# ------------------------------------------
# STEP 3 : Performance Boost
# ------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



# ------------------------------------------
# STEP 4 : Build CNN Model
# ------------------------------------------
model = tf.keras.Sequential([

    tf.keras.layers.Rescaling(1./255, input_shape=(224,224,3)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])


# ------------------------------------------
# STEP 5 : Compile Model
# ------------------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ------------------------------------------
# STEP 6 : Train Model
# ------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)


# ------------------------------------------
# STEP 7 : Accuracy Graph
# ------------------------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("CNN Accuracy")
plt.show()


# ------------------------------------------
# STEP 8 : Evaluation Matrix
# ------------------------------------------
y_true = np.concatenate([y for x, y in val_ds], axis=0)

pred = model.predict(val_ds)
y_pred = np.argmax(pred, axis=1)

print(confusion_matrix(y_true, y_pred))

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))



# ------------------------------------------
# STEP 9 : Save Model
# ------------------------------------------
model.save("cnn_steel_defect_model.h5")
print("Model Saved Successfully")


# ==========================================
# NEXT CODE: MobileNetV2 for Grad-CAM
# ==========================================
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ------------------------------------------
# STEP 1 : Load Pretrained MobileNetV2
# ------------------------------------------
mobilenet = MobileNetV2(
    weights='imagenet',
    include_top=True
)


# ------------------------------------------
# STEP 2 : Image Path
# ------------------------------------------
img_path = r"D:\CP2\NEU-CLS\Scratches\Sc_150.bmp"


# ------------------------------------------
# STEP 3 : Load Image
# ------------------------------------------
img = tf.keras.utils.load_img(img_path, target_size=(224,224))
img_array = tf.keras.utils.img_to_array(img)

orig = img_array.astype("uint8")

img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)



# ------------------------------------------
# STEP 4 : Build Grad-CAM Model
# ------------------------------------------
last_conv_layer = "block_13_expand_relu"

grad_model = tf.keras.models.Model(
    inputs=mobilenet.inputs,
    outputs=[
        mobilenet.get_layer(last_conv_layer).output,
        mobilenet.output
    ]
)


# ------------------------------------------
# STEP 5 : Generate Heatmap
# ------------------------------------------
with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(img_array)

    pred_index = tf.argmax(predictions[0])

    loss = predictions[:, pred_index]

grads = tape.gradient(loss, conv_outputs)

weights = tf.reduce_mean(grads, axis=(0,1,2))

cam = tf.reduce_sum(weights * conv_outputs[0], axis=-1)

heatmap = np.maximum(cam, 0)
heatmap = heatmap / (np.max(heatmap) + 1e-8)

heatmap = cv2.resize(heatmap, (224,224))



# ------------------------------------------
# STEP 7 : Show Heatmap
# ------------------------------------------
plt.figure(figsize=(7,7))
plt.imshow(heatmap, cmap='jet')
plt.title("MobileNetV2 Grad-CAM")
plt.axis("off")
plt.show()



img_path = r"D:\CP2\NEU-CLS\Scratches\Sc_150.bmp"

img = tf.keras.utils.load_img(img_path, target_size=(224,224))

plt.figure(figsize=(7,7))
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")
plt.show()