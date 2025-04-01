"""
Script to train a basic CNN model for tumor detection.
This is a simplified model for demonstration purposes.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from ...config import AI_MODEL_PATH

def create_model():
    """Create a simple CNN model."""
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
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

def generate_synthetic_data(num_samples=1000):
    """Generate synthetic data for training."""
    # Create synthetic images (random noise with different patterns)
    X = np.random.randn(num_samples, 128, 128, 1)
    
    # Create synthetic labels (random binary classification)
    y = np.random.randint(0, 2, num_samples)
    
    # Add some patterns to make it more realistic
    for i in range(num_samples):
        if y[i] == 1:  # Tumor case
            # Add a circular pattern
            center_x = np.random.randint(30, 98)
            center_y = np.random.randint(30, 98)
            radius = np.random.randint(5, 15)
            y_coords, x_coords = np.ogrid[-center_y:128-center_y, -center_x:128-center_x]
            mask = x_coords*x_coords + y_coords*y_coords <= radius*radius
            X[i, mask] = np.random.normal(2, 0.5, mask.sum())
    
    return X, y

def train_model():
    """Train the model on synthetic data."""
    # Create model
    model = create_model()
    
    # Generate synthetic data
    X_train, y_train = generate_synthetic_data(1000)
    X_val, y_val = generate_synthetic_data(200)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Train the model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Save the model
    if not os.path.exists(AI_MODEL_PATH):
        os.makedirs(AI_MODEL_PATH)
    model.save(os.path.join(AI_MODEL_PATH, 'tumor_detection.h5'))
    
    print("Model trained and saved successfully!")
    print(f"Model saved to: {os.path.join(AI_MODEL_PATH, 'tumor_detection.h5')}")

if __name__ == "__main__":
    train_model() 