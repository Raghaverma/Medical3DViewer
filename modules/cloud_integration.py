"""
Cloud integration module for handling file uploads to AWS S3 and Firebase Storage.
"""

import os
import logging
from typing import Optional
import boto3
from botocore.exceptions import ClientError
import firebase_admin
from firebase_admin import storage, credentials
from config import AWS_CONFIG, FIREBASE_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

class CloudUploadError(Exception):
    """Custom exception for cloud upload errors."""
    pass

def initialize_cloud_services() -> None:
    """Initialize cloud services with proper credentials."""
    try:
        # Initialize AWS S3 client
        s3 = boto3.client(
            "s3",
            region_name=AWS_CONFIG["region_name"],
            endpoint_url=AWS_CONFIG["endpoint_url"]
        )
        
        # Initialize Firebase
        if not firebase_admin._apps:
            cred = credentials.Certificate("path/to/serviceAccountKey.json")
            firebase_admin.initialize_app(cred, {
                "storageBucket": FIREBASE_CONFIG["storage_bucket"],
                "databaseURL": FIREBASE_CONFIG["database_url"]
            })
            
        logger.info("Cloud services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize cloud services: {str(e)}")
        raise CloudUploadError(f"Cloud service initialization failed: {str(e)}")

def upload_to_s3(file_path: str, object_name: Optional[str] = None) -> str:
    """
    Upload a file to AWS S3.
    
    Args:
        file_path: Path to the file to upload
        object_name: Optional S3 object name. If not provided, uses the file name
        
    Returns:
        str: The public URL of the uploaded file
        
    Raises:
        CloudUploadError: If upload fails
    """
    if not os.path.exists(file_path):
        raise CloudUploadError(f"File not found: {file_path}")
        
    try:
        s3 = boto3.client("s3")
        bucket_name = AWS_CONFIG["bucket_name"]
        
        if object_name is None:
            object_name = os.path.basename(file_path)
            
        logger.info(f"Uploading {file_path} to S3 bucket {bucket_name}")
        s3.upload_file(file_path, bucket_name, object_name)
        
        url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        logger.info(f"Successfully uploaded to S3: {url}")
        return url
        
    except ClientError as e:
        error_msg = f"Failed to upload to S3: {str(e)}"
        logger.error(error_msg)
        raise CloudUploadError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error during S3 upload: {str(e)}"
        logger.error(error_msg)
        raise CloudUploadError(error_msg)

def upload_to_firebase(file_path: str, object_name: Optional[str] = None) -> str:
    """
    Upload a file to Firebase Storage.
    
    Args:
        file_path: Path to the file to upload
        object_name: Optional storage object name. If not provided, uses the file name
        
    Returns:
        str: The public URL of the uploaded file
        
    Raises:
        CloudUploadError: If upload fails
    """
    if not os.path.exists(file_path):
        raise CloudUploadError(f"File not found: {file_path}")
        
    try:
        bucket = storage.bucket(FIREBASE_CONFIG["storage_bucket"])
        
        if object_name is None:
            object_name = os.path.basename(file_path)
            
        blob = bucket.blob(object_name)
        logger.info(f"Uploading {file_path} to Firebase Storage")
        
        blob.upload_from_filename(file_path)
        url = blob.public_url
        
        logger.info(f"Successfully uploaded to Firebase: {url}")
        return url
        
    except Exception as e:
        error_msg = f"Failed to upload to Firebase: {str(e)}"
        logger.error(error_msg)
        raise CloudUploadError(error_msg)

def delete_from_s3(object_name: str) -> None:
    """
    Delete a file from AWS S3.
    
    Args:
        object_name: Name of the object to delete
        
    Raises:
        CloudUploadError: If deletion fails
    """
    try:
        s3 = boto3.client("s3")
        bucket_name = AWS_CONFIG["bucket_name"]
        
        logger.info(f"Deleting {object_name} from S3 bucket {bucket_name}")
        s3.delete_object(Bucket=bucket_name, Key=object_name)
        logger.info(f"Successfully deleted from S3: {object_name}")
        
    except Exception as e:
        error_msg = f"Failed to delete from S3: {str(e)}"
        logger.error(error_msg)
        raise CloudUploadError(error_msg)

def delete_from_firebase(object_name: str) -> None:
    """
    Delete a file from Firebase Storage.
    
    Args:
        object_name: Name of the object to delete
        
    Raises:
        CloudUploadError: If deletion fails
    """
    try:
        bucket = storage.bucket(FIREBASE_CONFIG["storage_bucket"])
        blob = bucket.blob(object_name)
        
        logger.info(f"Deleting {object_name} from Firebase Storage")
        blob.delete()
        logger.info(f"Successfully deleted from Firebase: {object_name}")
        
    except Exception as e:
        error_msg = f"Failed to delete from Firebase: {str(e)}"
        logger.error(error_msg)
        raise CloudUploadError(error_msg)
