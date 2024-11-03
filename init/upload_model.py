import os
import sys
from minio import Minio
from minio.error import S3Error

# MinIO client configuration
minio_client = Minio(
    "minio:9000",
    access_key="omareweis",
    secret_key="ommaha260801",
    secure=False
)

bucket_name = "models"

# Check if bucket exists, create if not
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)

# List of model files to upload
model_files = [
    "finetuned_skipgram_model.pth",
    "model_2_checkpoint_epoch_6.pth"
]

# Upload models to MinIO
for model_file in model_files:
    try:
        if os.path.exists(model_file):
            minio_client.fput_object(bucket_name, model_file, model_file)
            print(f"Uploaded {model_file} to {bucket_name}.")
        else:
            print(f"File {model_file} does not exist.")
    except S3Error as err:
        print(f"Error occurred: {err}", file=sys.stderr)
