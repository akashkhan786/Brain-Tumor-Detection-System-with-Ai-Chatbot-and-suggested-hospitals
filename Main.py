# from flask import Flask , render_template , request , send_from_directory
# from tensorflow.keras.models import load_model
# from keras.preprocessing.image import load_img , img_to_array
# import numpy as np
# import os
#
# # Create a Flask app instance
# app = Flask(__name__)
#
# model = load_model('Model/best_model.keras' , compile=False )
#
# # class
# class_labels = ['glioma','notumor','meningioma','pituitary']
#
# # Upload folder for images
# UPLOAD_FOLDER = './uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
#
# # predict function
# def predict(image_path):
#     Image_size = 128
#     img = load_img(image_path, target_size=(Image_size, Image_size))
#     img_array = img_to_array(img) / 255.0  # Normalization
#     img_array = np.expand_dims(img_array, axis=0)  #batch dimension
#
#     prediction = model.predict(img_array)
#     predicted_class_index = np.argmax(prediction , axis = 1)[0]
#     confidence_score = np.max(prediction , axis = 1)[0]
#     if class_labels[predicted_class_index] == 'No tumor':
#         return "No tumor", confidence_score
#     else:
#         return f"Tumor: {class_labels[predicted_class_index]}", confidence_score
#
# # Routes
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # File Uploade
#         file = request.files['file']
#         if file:
#             file_location = os.path.join((UPLOAD_FOLDER),file.filename)
#             file.save(file_location)
#             print(f"File saved to: {file_location}")
#             # predict result
#             result , confidence = predict(file_location)
#             return render_template('index.html', result=result, confidence=f'{confidence*100:.2f}%' ,file_path= f'/uploads/{file.filename}' )
#     return render_template('index.html', result=None )
#
# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)
# /////////////// gpt //////////////////////

from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Create a Flask app instance
app = Flask(__name__)

model = load_model('Model/best_model.keras', compile=False)

# Class labels
class_labels = ['glioma', 'notumor', 'meningioma', 'pituitary']

# Upload folder for images
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Predict function
def predict(image_path):
    Image_size = 128
    img = load_img(image_path, target_size=(Image_size, Image_size))
    img_array = img_to_array(img) / 255.0  # Normalization
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # File upload
        file = request.files['file']
        if file:
            file_location = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_location)
            print(f"File saved to: {file_location}")
            # Predict result
            result, confidence = predict(file_location)
            return render_template('index.html', result=result, confidence=f'{confidence*100:.2f}%', file_path=f'/uploads/{file.filename}')
    return render_template('index.html', result=None)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)













