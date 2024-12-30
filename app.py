from __future__ import division, print_function
# coding=utf-8

import os
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image

# Define a Flask app
app = Flask(__name__)

# Path to the saved YOLOv8 model
MODEL_PATH = 'best.pt'

# Load your trained YOLOv8 model using the Ultralytics library
model = YOLO(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')

# Function to resize the image to 640x640
def resize_image(image_path, output_size=(640, 640)):
    with Image.open(image_path) as img:
        img_resized = img.resize(output_size)  # Resize to 640x640
        img_resized.save(image_path)  # Overwrite the file with resized image

# Function for processing the input image and prediction
def model_predict(img_path, model):
    # Resize the image to 640x640 before prediction
    resize_image(img_path)

    # Use the YOLOv8 model's predict function directly
    results = model(img_path)  # Perform inference on the input image

    # Extract the class with the highest confidence
    # (Assumes there is at least one detection; handle no-detection cases if needed)
    if len(results[0].boxes) > 0:
        # Extract class index of the most confident detection
        pred_class = int(results[0].boxes[0].cls.cpu().numpy())
    else:
        # Handle no detections
        pred_class = None
    return pred_class


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)  # Ensure the directory exists

        file_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        dic = {0: "Detected Trees"}  # Update this dictionary to match your model's labels
        if preds is not None:
            pred_class = dic.get(preds, "Unknown")  # Map the prediction to a human-readable label
        else:
            pred_class = "No detection"  # Handle case where no detections are made
        return pred_class
    return None


if __name__ == '__main__':
    app.run(debug=False)
