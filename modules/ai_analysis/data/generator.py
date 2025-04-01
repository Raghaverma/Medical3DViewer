"""
Data generation utilities for AI analysis.
"""

import numpy as np
from typing import Tuple, Optional

def generate_synthetic_data(
    num_samples: int = 1000,
    image_size: Tuple[int, int] = (128, 128),
    num_channels: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for training.
    
    Args:
        num_samples: Number of samples to generate
        image_size: Size of each image
        num_channels: Number of channels in each image
        
    Returns:
        Tuple of (X, y) where X is the image data and y is the labels
    """
    # Create synthetic images (random noise with different patterns)
    X = np.random.randn(num_samples, *image_size, num_channels)
    
    # Create synthetic labels (random binary classification)
    y = np.random.randint(0, 2, num_samples)
    
    # Add some patterns to make it more realistic
    for i in range(num_samples):
        if y[i] == 1:  # Tumor case
            # Add a circular pattern
            center_x = np.random.randint(30, image_size[0]-30)
            center_y = np.random.randint(30, image_size[1]-30)
            radius = np.random.randint(5, 15)
            y_coords, x_coords = np.ogrid[-center_y:image_size[1]-center_y, 
                                         -center_x:image_size[0]-center_x]
            mask = x_coords*x_coords + y_coords*y_coords <= radius*radius
            X[i, mask] = np.random.normal(2, 0.5, mask.sum())
    
    return X, y

def generate_segmentation_data(
    num_samples: int = 1000,
    image_size: Tuple[int, int] = (128, 128),
    num_channels: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for segmentation training.
    
    Args:
        num_samples: Number of samples to generate
        image_size: Size of each image
        num_channels: Number of channels in each image
        
    Returns:
        Tuple of (X, y) where X is the image data and y is the segmentation masks
    """
    # Create synthetic images
    X = np.random.randn(num_samples, *image_size, num_channels)
    
    # Create synthetic masks
    y = np.zeros((num_samples, *image_size, 1))
    
    # Add random shapes to masks
    for i in range(num_samples):
        # Random number of shapes
        num_shapes = np.random.randint(1, 4)
        
        for _ in range(num_shapes):
            # Random shape type
            shape_type = np.random.choice(['circle', 'rectangle'])
            
            if shape_type == 'circle':
                # Add circle
                center_x = np.random.randint(30, image_size[0]-30)
                center_y = np.random.randint(30, image_size[1]-30)
                radius = np.random.randint(5, 15)
                y_coords, x_coords = np.ogrid[-center_y:image_size[1]-center_y, 
                                             -center_x:image_size[0]-center_x]
                mask = x_coords*x_coords + y_coords*y_coords <= radius*radius
                y[i, mask] = 1
                
            else:
                # Add rectangle
                x1 = np.random.randint(20, image_size[0]-20)
                y1 = np.random.randint(20, image_size[1]-20)
                width = np.random.randint(10, 30)
                height = np.random.randint(10, 30)
                y[i, y1:y1+height, x1:x1+width] = 1
    
    return X, y

def generate_landmark_data(
    num_samples: int = 1000,
    image_size: Tuple[int, int] = (128, 128),
    num_channels: int = 1,
    num_landmarks: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for landmark detection training.
    
    Args:
        num_samples: Number of samples to generate
        image_size: Size of each image
        num_channels: Number of channels in each image
        num_landmarks: Number of landmarks to generate per image
        
    Returns:
        Tuple of (X, y) where X is the image data and y is the landmark coordinates
    """
    # Create synthetic images
    X = np.random.randn(num_samples, *image_size, num_channels)
    
    # Create synthetic landmark coordinates
    y = np.zeros((num_samples, num_landmarks * 2))  # x,y coordinates for each landmark
    
    # Add random landmarks
    for i in range(num_samples):
        for j in range(num_landmarks):
            # Generate random x,y coordinates
            x = np.random.randint(0, image_size[0])
            y[i, j*2] = x / image_size[0]  # Normalize coordinates
            y[i, j*2+1] = np.random.randint(0, image_size[1]) / image_size[1]
            
            # Add a marker at the landmark location
            marker_size = 3
            y_coords, x_coords = np.ogrid[-y[i, j*2+1]*image_size[1]:image_size[1]-y[i, j*2+1]*image_size[1],
                                         -x:image_size[0]-x]
            mask = (x_coords*x_coords + y_coords*y_coords <= marker_size*marker_size)
            X[i, mask] = np.random.normal(2, 0.5, mask.sum())
    
    return X, y 