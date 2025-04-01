"""
Model architectures for AI analysis.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def create_tumor_detection_model(input_shape=(128, 128, 1)):
    """
    Create a CNN model for tumor detection.
    
    Args:
        input_shape: Shape of input images
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_segmentation_model(input_shape=(128, 128, 1)):
    """
    Create a U-Net model for anatomical segmentation.
    
    Args:
        input_shape: Shape of input images
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    # Bridge
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    
    # Decoder
    up1 = layers.UpSampling2D((2, 2))(conv3)
    up1 = layers.concatenate([up1, conv2])
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    
    up2 = layers.UpSampling2D((2, 2))(conv4)
    up2 = layers.concatenate([up2, conv1])
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    
    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_landmark_detection_model(input_shape=(128, 128, 1)):
    """
    Create a model for anatomical landmark detection.
    
    Args:
        input_shape: Shape of input images
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Feature extraction
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(20, activation='sigmoid')  # 10 landmarks, each with x,y coordinates
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model 