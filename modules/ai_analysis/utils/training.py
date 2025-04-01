"""
Training utilities for AI analysis.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    epochs: int = 100,
    validation_split: float = 0.2,
    class_weights: Optional[Dict[int, float]] = None,
    callbacks: Optional[list] = None
) -> Dict[str, Any]:
    """
    Train a model with standard callbacks and monitoring.
    
    Args:
        model: Keras model to train
        X_train: Training data
        y_train: Training labels
        X_val: Optional validation data
        y_val: Optional validation labels
        batch_size: Batch size for training
        epochs: Number of epochs to train
        validation_split: Fraction of training data to use for validation
        class_weights: Optional class weights for imbalanced data
        callbacks: Optional list of additional callbacks
        
    Returns:
        Dictionary containing training history and model information
    """
    # Set up default callbacks
    default_callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Add any additional callbacks
    if callbacks:
        default_callbacks.extend(callbacks)
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val) if X_val is not None else None,
        validation_split=validation_split if X_val is None else None,
        class_weight=class_weights,
        callbacks=default_callbacks,
        verbose=1
    )
    
    return {
        'history': history.history,
        'model': model,
        'best_epoch': np.argmin(history.history['val_loss']) + 1,
        'best_loss': np.min(history.history['val_loss'])
    }

def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: Test labels
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Evaluate the model
    loss, *metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    
    # Get predictions
    y_pred = model.predict(X_test, batch_size=batch_size)
    
    # Calculate additional metrics
    results = {
        'loss': loss,
        'accuracy': metrics[0] if len(metrics) > 0 else None
    }
    
    # Add task-specific metrics
    if model.output_shape[-1] == 1:  # Binary classification
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred > 0.5, average='binary'
        )
        results.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return results

def predict_batch(
    model: Model,
    X: np.ndarray,
    batch_size: int = 32,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Make predictions on a batch of data.
    
    Args:
        model: Trained Keras model
        X: Input data
        batch_size: Batch size for prediction
        threshold: Classification threshold for binary tasks
        
    Returns:
        Model predictions
    """
    predictions = model.predict(X, batch_size=batch_size)
    
    # Apply threshold for binary classification
    if model.output_shape[-1] == 1:
        predictions = (predictions > threshold).astype(int)
    
    return predictions 