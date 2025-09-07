import os
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify
from model import CropDiseaseModel
from preprocessing import preprocess_image
from inference import generate_prediction

app = Flask(__name__)

# Load disease metadata
with open('assets/disease_metadata.json') as f:
    DISEASE_METADATA = json.load(f)

# Initialize model
model = CropDiseaseModel()
model.load_weights('assets/model.tflite')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Generate prediction
        prediction = generate_prediction(model, processed_image)
        
        # Apply confidence threshold
        if prediction['confidence'] < 0.7:
            return jsonify({
                'status': 'low_confidence',
                'message': 'Disease detection confidence too low. Please retake photo.'
            })
            
        # Get treatment recommendations
        disease_id = prediction['disease_id'] 
        recommendations = DISEASE_METADATA[disease_id]['recommendations']
        
        return jsonify({
            'status': 'success',
            'disease': prediction['disease_name'],
            'confidence': prediction['confidence'],
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)