"""
Configuration settings for the Medical 3D Viewer application.
"""

import os
from typing import Dict, Any

# Application settings
APP_NAME = "Medical 3D Viewer"
APP_VERSION = "1.0.0"
WINDOW_TITLE = f"{APP_NAME} v{APP_VERSION}"
WINDOW_GEOMETRY = (100, 100, 1000, 700)

# File paths
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")

# Supported file formats
SUPPORTED_3D_FORMATS = ["*.stl", "*.obj"]
SUPPORTED_MEDICAL_FORMATS = ["*.dcm", "*.nii", "*.nii.gz"]
ALL_SUPPORTED_FORMATS = SUPPORTED_3D_FORMATS + SUPPORTED_MEDICAL_FORMATS

# Cloud storage settings
AWS_CONFIG: Dict[str, Any] = {
    "region_name": "us-east-1",
    "bucket_name": "your-s3-bucket-name",
    "endpoint_url": None,
}

FIREBASE_CONFIG: Dict[str, Any] = {
    "project_id": "your-project-id",
    "storage_bucket": "your-firebase-bucket-name",
    "database_url": "https://your-project-id.firebaseio.com",
}

# Visualization settings
DEFAULT_BACKGROUND_COLOR = (0.2, 0.3, 0.4)  # Dark blue
DEFAULT_AXES_COLOR = (1.0, 1.0, 1.0)  # White
DEFAULT_BOUNDING_BOX_COLOR = (0.8, 0.8, 0.8)  # Light gray

# AI Analysis settings
AI_MODEL_PATH = os.path.join(ASSETS_DIR, "models")
CONFIDENCE_THRESHOLD = 0.8

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = "medical_3d_viewer.log"

# Create necessary directories
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(AI_MODEL_PATH, exist_ok=True) 