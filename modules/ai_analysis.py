"""
AI analysis module for the Medical 3D Viewer application.
Provides functions for AI-powered analysis of medical images.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from config import AI_MODEL_PATH, CONFIDENCE_THRESHOLD

# Configure logging
logger = logging.getLogger(__name__)

class AIAnalysisError(Exception):
    """Custom exception for AI analysis errors."""
    pass

class AIModelManager:
    """Manager class for AI models."""
    
    def __init__(self):
        self.models: Dict[str, tf.keras.Model] = {}
        self.load_models()
        
    def load_models(self) -> None:
        """Load all AI models from the models directory."""
        try:
            if not os.path.exists(AI_MODEL_PATH):
                logger.warning(f"Models directory not found: {AI_MODEL_PATH}")
                return
                
            for model_file in os.listdir(AI_MODEL_PATH):
                if model_file.endswith('.h5'):
                    model_name = os.path.splitext(model_file)[0]
                    model_path = os.path.join(AI_MODEL_PATH, model_file)
                    
                    logger.info(f"Loading model: {model_name}")
                    self.models[model_name] = load_model(model_path)
                    
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise AIAnalysisError(f"Failed to load models: {str(e)}")
            
    def get_model(self, model_name: str) -> Optional[tf.keras.Model]:
        """Get a model by name."""
        return self.models.get(model_name)
        
    def predict(
        self,
        model_name: str,
        image: np.ndarray,
        preprocess: bool = True
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Run prediction using a specific model.
        
        Args:
            model_name: Name of the model to use
            image: Input image array
            preprocess: Whether to preprocess the image
            
        Returns:
            Tuple of (confidence, prediction details)
            
        Raises:
            AIAnalysisError: If prediction fails
        """
        model = self.get_model(model_name)
        if model is None:
            raise AIAnalysisError(f"Model not found: {model_name}")
            
        try:
            if preprocess:
                image = self.preprocess_image(image)
                
            prediction = model.predict(image, verbose=0)
            confidence = float(prediction[0][0])
            
            details = {
                "model": model_name,
                "confidence": confidence,
                "threshold": CONFIDENCE_THRESHOLD,
                "prediction": "positive" if confidence > CONFIDENCE_THRESHOLD else "negative"
            }
            
            return confidence, details
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise AIAnalysisError(f"Prediction failed: {str(e)}")
            
    @staticmethod
    def preprocess_image(
        image: np.ndarray,
        target_size: Tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """
        Preprocess an image for model input.
        
        Args:
            image: Input image array
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image array
        """
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=[0, -1])
        
        return image

# Create global model manager instance
model_manager = AIModelManager()

def analyze_dicom(
    image: np.ndarray,
    model_name: str = "tumor_detection",
    return_details: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Run AI analysis on a DICOM image slice.
    
    Args:
        image: Input DICOM image array
        model_name: Name of the model to use
        return_details: Whether to return detailed prediction information
        
    Returns:
        Analysis result string or detailed prediction information
        
    Raises:
        AIAnalysisError: If analysis fails
    """
    try:
        confidence, details = model_manager.predict(model_name, image)
        
        if return_details:
            return details
            
        if confidence > CONFIDENCE_THRESHOLD:
            return f"Possible Tumor Detected (Confidence: {confidence:.2f})"
        return f"No Tumor Detected (Confidence: {confidence:.2f})"
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise AIAnalysisError(f"Analysis failed: {str(e)}")

def analyze_volume(
    volume: np.ndarray,
    model_name: str = "tumor_detection",
    slice_interval: int = 5
) -> Dict[str, Any]:
    """
    Run AI analysis on a 3D volume.
    
    Args:
        volume: 3D volume array
        model_name: Name of the model to use
        slice_interval: Interval between analyzed slices
        
    Returns:
        Dictionary containing analysis results
        
    Raises:
        AIAnalysisError: If analysis fails
    """
    try:
        results = []
        num_slices = volume.shape[0]
        
        for i in range(0, num_slices, slice_interval):
            slice_data = volume[i]
            confidence, details = model_manager.predict(model_name, slice_data)
            
            if confidence > CONFIDENCE_THRESHOLD:
                results.append({
                    "slice_index": i,
                    "confidence": confidence,
                    "details": details
                })
                
        return {
            "total_slices": num_slices,
            "analyzed_slices": len(results),
            "positive_findings": len(results),
            "findings": results
        }
        
    except Exception as e:
        logger.error(f"Volume analysis failed: {str(e)}")
        raise AIAnalysisError(f"Volume analysis failed: {str(e)}")

def segment_anatomy(
    image: np.ndarray,
    model_name: str = "anatomy_segmentation"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Perform anatomical segmentation on an image.
    
    Args:
        image: Input image array
        model_name: Name of the segmentation model to use
        
    Returns:
        Tuple of (segmentation mask, segmentation details)
        
    Raises:
        AIAnalysisError: If segmentation fails
    """
    try:
        confidence, details = model_manager.predict(model_name, image)
        
        # Create binary mask from prediction
        mask = (confidence > CONFIDENCE_THRESHOLD).astype(np.uint8)
        
        # Calculate segmentation metrics
        area = np.sum(mask)
        perimeter = cv2.arcLength(mask.astype(np.uint8), True)
        
        details.update({
            "area": area,
            "perimeter": perimeter,
            "mask_shape": mask.shape
        })
        
        return mask, details
        
    except Exception as e:
        logger.error(f"Segmentation failed: {str(e)}")
        raise AIAnalysisError(f"Segmentation failed: {str(e)}")

def detect_landmarks(
    image: np.ndarray,
    model_name: str = "landmark_detection"
) -> List[Dict[str, Any]]:
    """
    Detect anatomical landmarks in an image.
    
    Args:
        image: Input image array
        model_name: Name of the landmark detection model to use
        
    Returns:
        List of detected landmarks with their coordinates
        
    Raises:
        AIAnalysisError: If landmark detection fails
    """
    try:
        confidence, details = model_manager.predict(model_name, image)
        
        # Process landmark predictions
        landmarks = []
        for i, (x, y) in enumerate(details.get("coordinates", [])):
            if details.get("confidences", [])[i] > CONFIDENCE_THRESHOLD:
                landmarks.append({
                    "index": i,
                    "x": x,
                    "y": y,
                    "confidence": details["confidences"][i],
                    "name": details.get("names", [])[i] if "names" in details else f"Landmark {i}"
                })
                
        return landmarks
        
    except Exception as e:
        logger.error(f"Landmark detection failed: {str(e)}")
        raise AIAnalysisError(f"Landmark detection failed: {str(e)}")
