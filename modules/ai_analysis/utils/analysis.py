"""
Analysis utilities for Medical 3D Viewer.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

def analyze_dicom(image: np.ndarray, model_path: Optional[str] = None) -> Tuple[bool, float]:
    """
    Analyze a DICOM image slice using a pre-trained model.
    
    Args:
        image: The input image slice as a numpy array
        model_path: Path to the pre-trained model. If None, uses default model.
    
    Returns:
        Tuple of (is_tumor_detected: bool, confidence: float)
    
    Raises:
        ValueError: If image is invalid or model cannot be loaded
    """
    try:
        # Default model path if not specified
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tumor_detection.h5')
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")
            
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image[np.newaxis, ..., np.newaxis])
        confidence = float(prediction[0][0])
        is_tumor = confidence > 0.5
        
        logger.info(f"Analysis completed. Tumor detected: {is_tumor}, Confidence: {confidence:.2f}")
        return is_tumor, confidence
        
    except Exception as e:
        logger.error(f"Error during DICOM analysis: {str(e)}")
        raise

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image for the model.
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Preprocessed image
    """
    # Resize to model input size
    target_size = (128, 128)
    resized = tf.image.resize(image, target_size)
    
    # Normalize pixel values
    normalized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized))
    
    return normalized 