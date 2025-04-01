"""
Tests for the AI analysis module.
"""

import os
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from modules.ai_analysis import (
    analyze_dicom,
    analyze_volume,
    segment_anatomy,
    detect_landmarks,
    AIAnalysisError,
    # Models
    create_tumor_detection_model,
    create_segmentation_model,
    create_landmark_detection_model,
    
    # Data Generation
    generate_synthetic_data,
    generate_segmentation_data,
    generate_landmark_data,
    
    # Training
    train_model,
    evaluate_model,
    predict_batch,
    
    # Preprocessing
    normalize_image,
    resize_image,
    enhance_image,
    extract_patches,
    augment_image
)
from config import AI_MODEL_PATH
import pytest

class TestAIAnalysis(unittest.TestCase):
    """Test cases for AI analysis functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = "test_data"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
            
    def tearDown(self):
        """Clean up test environment."""
        # Remove test files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)
        
    def test_analyze_dicom(self):
        """Test DICOM analysis functionality."""
        # Create a test image
        image = np.random.rand(128, 128)
        
        # Test with invalid image
        with self.assertRaises(ValueError):
            analyze_dicom(None)
            
        # Test with invalid model name
        with self.assertRaises(AIAnalysisError):
            analyze_dicom(image, model_name="nonexistent_model")
            
    def test_analyze_volume(self):
        """Test volume analysis functionality."""
        # Create a test volume
        volume = np.random.rand(10, 128, 128)
        
        # Test with invalid volume
        with self.assertRaises(ValueError):
            analyze_volume(None)
            
        # Test with invalid slice interval
        with self.assertRaises(ValueError):
            analyze_volume(volume, slice_interval=0)
            
    def test_segment_anatomy(self):
        """Test anatomical segmentation functionality."""
        # Create a test image
        image = np.random.rand(128, 128)
        
        # Test with invalid image
        with self.assertRaises(ValueError):
            segment_anatomy(None)
            
        # Test with invalid model name
        with self.assertRaises(AIAnalysisError):
            segment_anatomy(image, model_name="nonexistent_model")
            
    def test_detect_landmarks(self):
        """Test landmark detection functionality."""
        # Create a test image
        image = np.random.rand(128, 128)
        
        # Test with invalid image
        with self.assertRaises(ValueError):
            detect_landmarks(None)
            
        # Test with invalid model name
        with self.assertRaises(AIAnalysisError):
            detect_landmarks(image, model_name="nonexistent_model")
            
    def test_model_loading(self):
        """Test AI model loading."""
        # Test with nonexistent model
        model_path = os.path.join(AI_MODEL_PATH, "nonexistent_model.h5")
        with self.assertRaises(AIAnalysisError):
            load_model(model_path)

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.rand(64, 64, 1)

@pytest.fixture
def sample_batch():
    """Create a sample batch of images for testing."""
    return np.random.rand(10, 64, 64, 1)

def test_model_creation():
    """Test model creation functions."""
    # Test tumor detection model
    tumor_model = create_tumor_detection_model()
    assert tumor_model.input_shape == (None, 128, 128, 1)
    assert tumor_model.output_shape == (None, 1)
    
    # Test segmentation model
    seg_model = create_segmentation_model()
    assert seg_model.input_shape == (None, 128, 128, 1)
    assert seg_model.output_shape == (None, 128, 128, 1)
    
    # Test landmark detection model
    landmark_model = create_landmark_detection_model()
    assert landmark_model.input_shape == (None, 128, 128, 1)
    assert landmark_model.output_shape == (None, 20)  # 10 landmarks * 2 coordinates

def test_data_generation():
    """Test data generation functions."""
    # Test synthetic data generation
    X, y = generate_synthetic_data(num_samples=10)
    assert X.shape == (10, 128, 128, 1)
    assert y.shape == (10,)
    assert np.all(np.isin(y, [0, 1]))
    
    # Test segmentation data generation
    X, y = generate_segmentation_data(num_samples=10)
    assert X.shape == (10, 128, 128, 1)
    assert y.shape == (10, 128, 128, 1)
    assert np.all(np.isin(y, [0, 1]))
    
    # Test landmark data generation
    X, y = generate_landmark_data(num_samples=10)
    assert X.shape == (10, 128, 128, 1)
    assert y.shape == (10, 20)  # 10 landmarks * 2 coordinates
    assert np.all(y >= 0) and np.all(y <= 1)

def test_preprocessing(sample_image):
    """Test preprocessing functions."""
    # Test normalization
    normalized = normalize_image(sample_image)
    assert np.min(normalized) >= 0
    assert np.max(normalized) <= 1
    
    # Test resizing
    resized = resize_image(sample_image, (32, 32))
    assert resized.shape == (32, 32, 1)
    
    # Test enhancement
    enhanced = enhance_image(sample_image)
    assert enhanced.shape == sample_image.shape
    
    # Test patch extraction
    patches = extract_patches(sample_image, (16, 16))
    assert len(patches.shape) == 4  # (num_patches, height, width, channels)
    
    # Test augmentation
    augmented = augment_image(sample_image)
    assert augmented.shape == sample_image.shape

def test_training_and_evaluation(sample_batch):
    """Test training and evaluation functions."""
    # Create a simple model for testing
    model = create_tumor_detection_model()
    
    # Generate synthetic data
    X_train = sample_batch
    y_train = np.random.randint(0, 2, 10)
    
    # Test training
    history = train_model(
        model,
        X_train,
        y_train,
        epochs=2,
        batch_size=4
    )
    assert 'history' in history
    assert 'model' in history
    assert 'best_epoch' in history
    assert 'best_loss' in history
    
    # Test evaluation
    metrics = evaluate_model(model, X_train, y_train)
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    
    # Test prediction
    predictions = predict_batch(model, X_train)
    assert predictions.shape == (10, 1)
    assert np.all(np.isin(predictions, [0, 1]))

def test_error_handling():
    """Test error handling in various functions."""
    # Test invalid input shapes
    with pytest.raises(ValueError):
        normalize_image(np.random.rand(64, 64))  # Missing channel dimension
    
    # Test invalid patch sizes
    with pytest.raises(ValueError):
        extract_patches(np.random.rand(64, 64, 1), (32, 32, 1))  # Invalid patch shape
    
    # Test invalid model inputs
    model = create_tumor_detection_model()
    with pytest.raises(ValueError):
        predict_batch(model, np.random.rand(10, 64, 64))  # Missing channel dimension

if __name__ == '__main__':
    unittest.main() 