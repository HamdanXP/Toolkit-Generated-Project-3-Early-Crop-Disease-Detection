import numpy as np

def generate_prediction(model, image):
    """Generate disease prediction for input image
    
    Args:
        model: Loaded model instance
        image: Preprocessed image array
        
    Returns:
        Dictionary containing prediction details
    """
    # Get model prediction
    prediction = model.predict(image)
    
    # Get predicted class
    class_id = np.argmax(prediction)
    confidence = float(prediction[0][class_id])
    
    # Load class labels
    with open('assets/labels.txt') as f:
        labels = f.read().splitlines()
    
    return {
        'disease_id': class_id,
        'disease_name': labels[class_id],
        'confidence': confidence
    }