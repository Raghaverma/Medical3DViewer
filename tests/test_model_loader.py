"""
Tests for the model loader module.
"""

import os
import unittest
import numpy as np
from vtk import vtkActor
from modules.model_loader import (
    load_model,
    get_model_info,
    center_model,
    normalize_model,
    ModelLoadError
)

class TestModelLoader(unittest.TestCase):
    """Test cases for model loader functionality."""
    
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
        
    def test_load_model(self):
        """Test model loading functionality."""
        # Test with invalid file
        with self.assertRaises(ModelLoadError):
            load_model("nonexistent.stl")
            
        # Test with unsupported format
        with self.assertRaises(ModelLoadError):
            load_model("test.txt")
            
    def test_get_model_info(self):
        """Test model information retrieval."""
        # Create a test actor
        actor = vtkActor()
        
        # Test with invalid actor
        with self.assertRaises(ValueError):
            get_model_info(None)
            
    def test_center_model(self):
        """Test model centering functionality."""
        # Create a test actor
        actor = vtkActor()
        
        # Test with invalid actor
        with self.assertRaises(ValueError):
            center_model(None)
            
    def test_normalize_model(self):
        """Test model normalization functionality."""
        # Create a test actor
        actor = vtkActor()
        
        # Test with invalid actor
        with self.assertRaises(ValueError):
            normalize_model(None)
            
    def test_model_parameters(self):
        """Test model parameter validation."""
        # Test invalid color
        with self.assertRaises(ValueError):
            load_model(
                "test.stl",
                color=(-1, 0, 0)  # Invalid color value
            )
            
        # Test invalid opacity
        with self.assertRaises(ValueError):
            load_model(
                "test.stl",
                opacity=2.0  # Invalid opacity value
            )
            
        # Test invalid edge width
        with self.assertRaises(ValueError):
            load_model(
                "test.stl",
                edge_width=-1  # Invalid edge width
            )

if __name__ == '__main__':
    unittest.main() 