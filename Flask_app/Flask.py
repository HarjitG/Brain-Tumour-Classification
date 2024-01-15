import flask
import tensorflow as tf
import numpy as np

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('Flask_app/model')

global Class_Names

Class_Names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Upload', methods = ['POST'])
def Upload():
    if 'file' not in request.files:
        return render_template('index.html', message='No file path')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No selected file')

    try:
        img_path = 'Flask_app/uploads/' + file.filename
        file.save(img_path)
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        class_label = np.argmax(prediction[0])
        label = Class_Names[class_label].replace('\'', '')
        
       # class_label = np.argmax(prediction[0])
        result = f'Tumor Class: {label}'
    except Exception as e:
        result = f'Error: {e}'

    return render_template('index.html', message=result)


if __name__ == "__main__":
    app.run()