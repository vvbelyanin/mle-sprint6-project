# Python file: s3_utils.py
# 
# Content:
# 1. Variables/Constants:
#    - S3_BUCKET_NAME: The name of the S3 bucket where files are stored.
#    - AWS_ACCESS_KEY_ID: AWS access key ID used for authentication.
#    - AWS_SECRET_ACCESS_KEY: AWS secret access key for authentication.
#    - S3_ENDPOINT_URL: The endpoint URL for connecting to S3.
#    - AWS_CONN_ID: The connection ID for AWS (used by Airflow).
#
# 2. Functions:
#    - get_s3_client(): Initializes and returns an S3 client for interacting with AWS S3.
#    - download_parquet_from_s3(): Downloads a Parquet file from S3 and loads it into a Pandas DataFrame.
#    - upload_dataframe_to_s3_as_parquet(): Uploads a Pandas DataFrame as a Parquet file to S3.
#    - show_s3_folder(): Lists all files in a specified S3 folder.
#    - download_from_s3(): Downloads a file from S3 and returns it as a Pandas DataFrame or CatBoost model.
#    - upload_to_s3(): Uploads a file to S3, returning True if successful.
#
# This file contains utility functions to handle interaction with AWS S3 using boto3, Airflow hooks, and Pandas.
# It covers common tasks such as uploading, downloading, and listing files in S3.

# Boto3 library to interact with AWS S3
import boto3

# Exceptions related to AWS operations for better error handling
from botocore.exceptions import ClientError, PartialCredentialsError

# S3Hook from Airflow to interface with AWS S3 storage in workflows
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# Standard library imports for file operations and typing
import os
from typing import Optional, Any

# CatBoostClassifier for loading CatBoost models from S3
from catboost import CatBoostClassifier

# Pandas for data manipulation and handling of parquet files
import pandas as pd

# Importing environment-specific configurations such as AWS credentials and S3 bucket details
from utils.config import (
    S3_BUCKET_NAME, 
    AWS_ACCESS_KEY_ID, 
    AWS_SECRET_ACCESS_KEY, 
    S3_ENDPOINT_URL, 
    AWS_CONN_ID
)


def get_s3_client() -> boto3.client:
    """
    Create and return an S3 client using boto3 with the specified credentials.

    Returns:
        boto3.client: A client object for interacting with S3.
    """
    return boto3.client(
        's3', 
        aws_access_key_id=AWS_ACCESS_KEY_ID, 
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        endpoint_url=S3_ENDPOINT_URL
    )


def download_parquet_from_s3(bucket_name: str, s3_key: str, local_path: str, aws_conn_id: str = AWS_CONN_ID) -> pd.DataFrame:
    """
    Download a parquet file from S3, load it into a Pandas DataFrame, and remove the local file.

    Args:
        bucket_name (str): The S3 bucket name.
        s3_key (str): The S3 object key (file path).
        local_path (str): The local path to save the file temporarily.
        aws_conn_id (str): The Airflow connection ID for AWS (defaults to AWS_CONN_ID).

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    s3_hook = S3Hook(aws_conn_id=aws_conn_id)
    s3_hook.get_key(key=s3_key, bucket_name=bucket_name).download_file(local_path)
    df = pd.read_parquet(local_path)
    os.remove(local_path)
    return df


def upload_dataframe_to_s3_as_parquet(df: pd.DataFrame, bucket_name: str, s3_key: str, local_path: str, aws_conn_id: str = AWS_CONN_ID) -> None:
    """
    Upload a Pandas DataFrame as a parquet file to S3.

    Args:
        df (pd.DataFrame): The DataFrame to upload.
        bucket_name (str): The S3 bucket name.
        s3_key (str): The S3 object key (file path).
        local_path (str): The local path to temporarily store the parquet file.
        aws_conn_id (str): The Airflow connection ID for AWS (defaults to AWS_CONN_ID).
    """
    df.to_parquet(local_path)
    s3_hook = S3Hook(aws_conn_id=aws_conn_id)
    s3_hook.load_file(filename=local_path, key=s3_key, bucket_name=bucket_name, replace=True)
    os.remove(local_path)


def show_s3_folder(path: str) -> None:
    """
    List all files in a specified S3 folder path.

    Args:
        path (str): The S3 folder path to list the contents from.
    """
    s3_client = get_s3_client()

    # Paginate through the files in the specified folder
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=path)

    print(f"Files in {path} folder:")
    counter = 0
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if not key.endswith('/'):
                    relative_path = key[len(path):]
                    if '/' not in relative_path:
                        print(key)
                        counter += 1
    if counter == 0:
        print("No files found.")
    else:
        print(f"{counter} file(s) found.")


def download_from_s3(file_name: str, object_name: Optional[str] = None) -> Any:
    """
    Download a file from an S3 bucket and return an appropriate object depending on the file type.

    Args:
        file_name (str): The name of the file to save locally.
        object_name (Optional[str]): The name of the object in the S3 bucket (defaults to file_name if not provided).

    Returns:
        Any: Loaded Parquet file or CatBoost model based on the file type.
    """
    s3_client = get_s3_client()
    object_name = object_name if object_name else file_name
    s3_client.download_file(S3_BUCKET_NAME, object_name, file_name)

    # Return the appropriate object based on the file extension
    if file_name.endswith('.parquet'):
        return pd.read_parquet(file_name)
    elif file_name.endswith('.cbm'):
        model = CatBoostClassifier()
        model.load_model(file_name)
        return model
    else:
        raise ValueError("Unsupported file type. Only .parquet and .cbm files are supported.")


def upload_to_s3(file_name: str, object_name: Optional[str] = None) -> bool:
    """
    Upload a file to an S3 bucket.

    Args:
        file_name (str): The local file path to upload.
        object_name (Optional[str]): The object name in the S3 bucket. Defaults to file_name if not provided.

    Returns:
        bool: True if the upload was successful, False otherwise.
    """
    s3_client = get_s3_client()
    object_name = object_name if object_name else file_name

    try:
        s3_client.upload_file(file_name, S3_BUCKET_NAME, object_name)
        print(f"File {file_name} uploaded to {object_name}")
        return True
    except PartialCredentialsError:
        print("Incomplete credentials provided.")
    except ClientError as e:
        print(f"Upload failed:\nDetails: {str(e)}")
    except Exception as e:
        print(f"General Error: An unexpected error occurred: {str(e)}")
    return False
