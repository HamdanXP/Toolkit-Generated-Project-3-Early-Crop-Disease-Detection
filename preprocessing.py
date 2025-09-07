import cv2
import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input
    
    Args:
        image: Input image array
        target_size: Model input size
        
    Returns:
        Preprocessed image array
    """
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def augment_image(image):
    """Apply data augmentation
    
    Args:
        image: Input image array
        
    Returns:
        Augmented image array
    """
    # Random rotation
    angle = np.random.uniform(-20, 20)
    image = Image.fromarray(image)
    image = image.rotate(angle)
    image = np.array(image)
    
    # Random brightness
    brightness = np.random.uniform(0.8, 1.2)
    image = image * brightness
    image = np.clip(image, 0, 255)
    
    return image.astype(np.uint8)