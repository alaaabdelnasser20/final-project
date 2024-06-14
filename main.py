from flask import Flask, request, jsonify, abort
import uuid
from datetime import datetime
import tensorflow as tf
from PIL import Image
import numpy as np
import base64

app = Flask(__name__)

model_path = 'C:\\Users\\ECC\\Desktop\\flaskapp\\flaskapp\\project\\tomato_model.h5'
model = tf.keras.models.load_model(model_path)


print(model)

# Define disease labels
disease_labels = [
    'Late blight',
    'Tomato Yellow Leaf Curl Virus',
    'Septoria leaf spot',
    'Early blight',
    'Spider mites Two-spotted spider mite',
    'Powdery mildew',
    'Healthy',
    'Bacterial spot',
    'Target Spot',
    'Tomato mosaic virus',
    'Leaf Mold'
]

@app.route('/api/process_image', methods=['POST'])
def process_image():
    # Receive the image file from the request
    if 'image' not in request.files:
        print('Image file not found')
        return jsonify({'error': 'Image file not found'}), 400

    image_file = request.files['image']
    print('Process the image using your AI model')
    # Process the image using your AI model
    try:
        print('Processing image...')
        prediction = process_image_with_model(image_file, model)
        
        # Map prediction to disease label
        predicted_disease = map_to_disease(prediction)

        # Return the predicted disease
        return jsonify({'predicted_disease': predicted_disease})
        
    except Exception as e:
        print(f'Image processing failed: {str(e)}')
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500

def process_image_with_model(image_file, model):
    # Open the image using Pillow
    image = Image.open(image_file)

    # Preprocess the image for the model
    processed_image = preprocess_image(image)

    # Perform inference using the model
    prediction = model.predict(processed_image)

    return prediction

def preprocess_image(image):
    # Resize, normalize, or apply any other preprocessing steps required by your model
    
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Preprocess the image array
    processed_array = tf.image.resize(image_array, [256, 256])
    processed_array = processed_array / 255.0

    # Expand the dimensions to match the model input shape
    processed_array = np.expand_dims(processed_array, axis=0)

    return processed_array

def map_to_disease(prediction):
    # Decode prediction and map to disease label
    # This function should map the prediction to the corresponding disease label
    # You can use argmax or any other logic based on your model's output
    predicted_index = np.argmax(prediction)
    predicted_disease = disease_labels[predicted_index]
    
    return predicted_disease

if __name__ == '__main__':
    app.run()
    
