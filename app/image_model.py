import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = os.path.join("model", "best_model.keras")
model = load_model(MODEL_PATH, compile=False)
class_labels = ['glioma', 'notumor', 'meningioma', 'pituitary']

def predict(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_labels[predicted_class], confidence
