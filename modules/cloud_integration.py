import boto3
import firebase_admin
from firebase_admin import storage

# AWS S3 Configuration
s3 = boto3.client("s3")

def upload_to_s3(file_path, bucket_name, object_name):
    """ Upload a file to AWS S3. """
    s3.upload_file(file_path, bucket_name, object_name)
    return f"https://{bucket_name}.s3.amazonaws.com/{object_name}"

# Firebase Configuration
firebase_admin.initialize_app()

def upload_to_firebase(file_path, bucket_name):
    """ Upload a file to Firebase Storage. """
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(file_path.split("/")[-1])
    blob.upload_from_filename(file_path)
    return blob.public_url
