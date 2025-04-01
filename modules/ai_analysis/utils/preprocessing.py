"""
Preprocessing utilities for AI analysis.
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy import ndimage
from skimage import exposure, filters

def normalize_image(
    image: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """
    Normalize image to specified range.
    
    Args:
        image: Input image
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
        
    Returns:
        Normalized image
    """
    # Convert to float
    image = image.astype(float)
    
    # Get current range
    current_min = np.min(image)
    current_max = np.max(image)
    
    # Avoid division by zero
    if current_max - current_min == 0:
        return np.full_like(image, min_val)
    
    # Normalize
    normalized = (image - current_min) / (current_max - current_min)
    return normalized * (max_val - min_val) + min_val

def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    preserve_aspect: bool = True
) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        preserve_aspect: Whether to preserve aspect ratio
        
    Returns:
        Resized image
    """
    if preserve_aspect:
        # Calculate scaling factor
        h, w = image.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_size = (int(h * scale), int(w * scale))
        
        # Resize
        resized = ndimage.zoom(image, (new_size[0] / h, new_size[1] / w))
        
        # Pad if necessary
        if new_size != target_size:
            padded = np.zeros((*target_size, *image.shape[2:]), dtype=image.dtype)
            y_offset = (target_size[0] - new_size[0]) // 2
            x_offset = (target_size[1] - new_size[1]) // 2
            padded[y_offset:y_offset+new_size[0], 
                  x_offset:x_offset+new_size[1]] = resized
            return padded
        return resized
    else:
        # Direct resize without preserving aspect ratio
        return ndimage.zoom(image, (target_size[0] / image.shape[0],
                                  target_size[1] / image.shape[1]))

def enhance_image(
    image: np.ndarray,
    contrast: bool = True,
    denoise: bool = True,
    sharpen: bool = True
) -> np.ndarray:
    """
    Enhance image quality.
    
    Args:
        image: Input image
        contrast: Whether to enhance contrast
        denoise: Whether to denoise
        sharpen: Whether to sharpen
        
    Returns:
        Enhanced image
    """
    enhanced = image.copy()
    
    if contrast:
        # Enhance contrast using histogram equalization
        enhanced = exposure.equalize_hist(enhanced)
    
    if denoise:
        # Apply Gaussian denoising
        enhanced = filters.gaussian(enhanced, sigma=1)
    
    if sharpen:
        # Apply unsharp masking
        enhanced = filters.unsharp_mask(enhanced, radius=1, amount=1)
    
    return enhanced

def extract_patches(
    image: np.ndarray,
    patch_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]] = None,
    padding: str = 'valid'
) -> np.ndarray:
    """
    Extract patches from image.
    
    Args:
        image: Input image
        patch_size: Size of patches to extract
        stride: Stride for patch extraction
        padding: Padding mode ('valid' or 'same')
        
    Returns:
        Array of patches
    """
    if stride is None:
        stride = patch_size
    
    h, w = image.shape[:2]
    ph, pw = patch_size
    sh, sw = stride
    
    if padding == 'valid':
        # Calculate number of patches
        n_h = (h - ph) // sh + 1
        n_w = (w - pw) // sw + 1
        
        # Initialize patches array
        patches = np.zeros((n_h * n_w, ph, pw, *image.shape[2:]), dtype=image.dtype)
        
        # Extract patches
        for i in range(n_h):
            for j in range(n_w):
                patches[i * n_w + j] = image[i*sh:i*sh+ph, j*sw:j*sw+pw]
    else:  # 'same' padding
        # Calculate padding
        pad_h = ((ph - 1) // 2, ph // 2)
        pad_w = ((pw - 1) // 2, pw // 2)
        
        # Pad image
        padded = np.pad(image, (pad_h, pad_w, *[(0, 0)] * (len(image.shape) - 2)))
        
        # Calculate number of patches
        n_h = h // sh
        n_w = w // sw
        
        # Initialize patches array
        patches = np.zeros((n_h * n_w, ph, pw, *image.shape[2:]), dtype=image.dtype)
        
        # Extract patches
        for i in range(n_h):
            for j in range(n_w):
                patches[i * n_w + j] = padded[i*sh:i*sh+ph, j*sw:j*sw+pw]
    
    return patches

def augment_image(
    image: np.ndarray,
    rotation_range: Tuple[float, float] = (-10, 10),
    shift_range: Tuple[float, float] = (-0.1, 0.1),
    zoom_range: Tuple[float, float] = (0.9, 1.1),
    flip_horizontal: bool = True,
    flip_vertical: bool = True,
    brightness_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """
    Apply random augmentations to image.
    
    Args:
        image: Input image
        rotation_range: Range for random rotation
        shift_range: Range for random shift
        zoom_range: Range for random zoom
        flip_horizontal: Whether to allow horizontal flips
        flip_vertical: Whether to allow vertical flips
        brightness_range: Range for random brightness adjustment
        
    Returns:
        Augmented image
    """
    augmented = image.copy()
    
    # Random rotation
    if rotation_range[0] != 0 or rotation_range[1] != 0:
        angle = np.random.uniform(*rotation_range)
        augmented = ndimage.rotate(augmented, angle, reshape=False)
    
    # Random shift
    if shift_range[0] != 0 or shift_range[1] != 0:
        shift = [np.random.uniform(*shift_range) * s for s in augmented.shape[:2]]
        augmented = ndimage.shift(augmented, shift)
    
    # Random zoom
    if zoom_range[0] != 1 or zoom_range[1] != 1:
        zoom = np.random.uniform(*zoom_range)
        augmented = ndimage.zoom(augmented, (zoom, zoom, *[1] * (len(augmented.shape) - 2)))
    
    # Random flips
    if flip_horizontal and np.random.random() < 0.5:
        augmented = np.fliplr(augmented)
    if flip_vertical and np.random.random() < 0.5:
        augmented = np.flipud(augmented)
    
    # Random brightness
    if brightness_range[0] != 1 or brightness_range[1] != 1:
        factor = np.random.uniform(*brightness_range)
        augmented = augmented * factor
    
    return augmented 