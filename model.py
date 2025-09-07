import tensorflow as tf
import numpy as np

class CropDiseaseModel:
    def __init__(self):
        self.model = None
        self.input_shape = (224, 224, 3)
        
    def build_model(self):
        """Build CNN model architecture optimized for mobile deployment"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model = model
        return model
    
    def load_weights(self, model_path):
        """Load trained model weights"""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        self.interpreter = interpreter
        
    def predict(self, image):
        """Generate prediction for input image"""
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.interpreter.set_tensor(input_details[0]['index'], image)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(output_details[0]['index'])
        
        return prediction