"""
Script to test the trained tumor detection model.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import AI_MODEL_PATH

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess an image for model input."""
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values
    image = image.astype('float32') / 255.0
    
    # Add batch and channel dimensions
    image = np.expand_dims(image, axis=[0, -1])
    
    return image

def generate_sample_image():
    """Generate a sample image for testing."""
    # Create a random image
    image = np.random.randn(128, 128)
    
    # Add a circular pattern (simulating a tumor)
    center_x = 64
    center_y = 64
    radius = 10
    y_coords, x_coords = np.ogrid[-center_y:128-center_y, -center_x:128-center_x]
    mask = x_coords*x_coords + y_coords*y_coords <= radius*radius
    image[mask] = np.random.normal(2, 0.5, mask.sum())
    
    # Normalize to 0-255 range
    image = ((image - image.min()) * (255 / (image.max() - image.min()))).astype(np.uint8)
    
    return image

def test_model():
    """Test the trained model."""
    # Load the model
    model_path = os.path.join(AI_MODEL_PATH, 'tumor_detection.h5')
    if not os.path.exists(model_path):
        print("Error: Model file not found!")
        print("Please train the model first using train_model.py")
        return
    
    model = load_model(model_path)
    
    # Generate and process a sample image
    image = generate_sample_image()
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)
    confidence = float(prediction[0][0])
    
    # Display results
    print("\nModel Test Results:")
    print("-" * 20)
    print(f"Confidence: {confidence:.2f}")
    print(f"Prediction: {'Tumor Detected' if confidence > 0.5 else 'No Tumor Detected'}")
    
    # Save the sample image
    cv2.imwrite('sample_image.png', image)
    print("\nSample image saved as 'sample_image.png'")

if __name__ == "__main__":
    test_model() 