# airflow/dags/pipeline.py
"""
Airflow DAG for downloading, processing, and uploading data to S3.

Content:
1. download_unzip_upload - Downloads a zip file, unzips it, and uploads the CSV to S3.
2. csv_extract - Extracts a CSV from S3, converts it to a Parquet file, and uploads it back to S3.
3. create_target_and_split - Processes the dataset to create targets, splits data into train and test, and uploads them to S3.
4. transform_fit - Loads and processes train/test data, fits a model, and uploads the model to S3.
"""

# Standard library imports
from datetime import datetime
import subprocess
import os

# Third-party library imports
import pandas as pd
import joblib
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Local application imports (helpers, configurations, and utilities)
from utils.helpers import get_mem_usage
from utils.s3 import download_parquet_from_s3, upload_dataframe_to_s3_as_parquet
from utils.config import (
    dtype_spec, new_columns, attrs, date_columns, path, 
    AWS_CONN_ID, DATA_CSV, DATA_ZIP, DATA_PARQUET, TMP_DIR, 
    S3_BUCKET_NAME, S3_DIR, TRAIN_TEST_SPLIT_DATE, TRAIN_PARQUET, 
    TEST_PARQUET, START_TRAIN_DATE, FITTED_MODEL
)
from utils.preprocess import process_na, model

with DAG(
    dag_id='sprint_6_airflow',
    default_args={'start_date': datetime(2024, 1, 1)},
    schedule=None,
    catchup=False
) as dag:
    
    def download_unzip_upload() -> None:
        """
        Downloads a zip file from Yandex Disk, unzips it, and uploads the CSV to S3.
        """
        print("Start of data extraction...")
        tmp_zip = path(TMP_DIR, DATA_ZIP)
        print(f"{tmp_zip=}")
        download_command = f'wget -O {tmp_zip} $(yadisk-direct https://disk.yandex.com/d/Io0siOESo2RAaA)'
        subprocess.run(download_command, shell=True, check=True)
        
        print("Start of unzip data...")
        tmp_csv = path(TMP_DIR, DATA_CSV)
        print(f"{tmp_csv=}")
        unzip_command = f'unzip -p {tmp_zip} train_ver2.csv > {tmp_csv}'
        subprocess.run(unzip_command, shell=True, check=True)
        
        print("Start of saving data to S3...")
        print(f"{S3_BUCKET_NAME=}")
        print(f"{S3_DIR=}")
        print(f"S3 Key: {path(S3_DIR, DATA_CSV)}")
        print(f"AWS_ACCESS_KEY_ID: {os.getenv('AWS_ACCESS_KEY_ID')}")
        print(f"{AWS_CONN_ID=}")
        
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        s3_hook.load_file(filename=tmp_csv, key=path(S3_DIR, DATA_CSV), bucket_name=S3_BUCKET_NAME, replace=True)
        
        print("Data was saved to S3.")
    
    def csv_extract() -> None:
        """
        Extracts a CSV from S3, converts it to a Parquet file, and uploads it back to S3.
        """
        print("Start of loading csv from S3...")
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        s3_object = s3_hook.get_key(key=path(S3_DIR, DATA_CSV), bucket_name=S3_BUCKET_NAME)
        s3_body = s3_object.get()['Body']
        
        print("Start of parsing csv...")
        df = pd.read_csv(
            s3_body, 
            dtype=dtype_spec,
            parse_dates=date_columns,
            date_format='%Y-%m-%d',
            low_memory=False,
            header=0,
            names=list(new_columns.values())
        )
        df.attrs = attrs
        tmp_parquet = path(TMP_DIR, DATA_PARQUET)
        
        print("Convert data to parquet...")
        df.to_parquet(tmp_parquet)
        
        print("Start of saving data to S3...")
        s3_hook.load_file(filename=tmp_parquet, key=path(S3_DIR, DATA_PARQUET), bucket_name=S3_BUCKET_NAME, replace=True)
        
        print("Data was saved to S3.")
        get_mem_usage()

    def create_target_and_split() -> None:
        """
        Processes the dataset to create target columns, splits data into train and test sets, and uploads them to S3.
        """
        print("Start of loading data from S3...")
        df = download_parquet_from_s3(S3_BUCKET_NAME, path(S3_DIR, DATA_PARQUET), path(TMP_DIR, DATA_PARQUET))

        print("Start of transforming DataFrame...")
        product_features = [col for col in df.columns if col.startswith('ind_1m_')]
        grouped = df.groupby('id')[product_features]

        # Shift product features by 1 period to create target variables
        shifted_values = grouped.shift(-1)
        df_targets = ((df[product_features] == 0) & (shifted_values == 1)).astype(int).add_prefix('target__')
        for col in df_targets.columns:
            df[col] = df_targets[col]
        del shifted_values, df_targets

        # Mask to remove the last row of each group (no future data for target creation)
        targets = [col for col in df.columns if col.startswith("target__")]
        last_row_mask = df.groupby('id')['id'].transform('shift', -1).notna()
        df = df[last_row_mask]

        # Filter out rows with no target events
        grouped_0 = (df.groupby('id')[targets].sum().sum(axis=1) == 0)
        df = df[~df.id.isin(grouped_0[grouped_0].index)]
        df.attrs['target__'] = targets
        get_mem_usage()

        print("Saving test dataset to S3...")
        # Split into train and test sets and upload to S3
        tmp_df_test = path(TMP_DIR, TEST_PARQUET)
        upload_dataframe_to_s3_as_parquet(
            df[df.fetch_date > TRAIN_TEST_SPLIT_DATE], 
            S3_BUCKET_NAME, 
            path(S3_DIR, TEST_PARQUET), 
            tmp_df_test
        )
        
        print("Saving train dataset to S3...")
        tmp_df_train = path(TMP_DIR, TRAIN_PARQUET)
        df = df[df.fetch_date <= TRAIN_TEST_SPLIT_DATE]
        upload_dataframe_to_s3_as_parquet(
            df, 
            S3_BUCKET_NAME, 
            path(S3_DIR, TRAIN_PARQUET), 
            tmp_df_train
        )
        print("Datasets were saved to S3.")

    def transform_fit() -> None:
        """
        Loads and processes train and test data, fits a machine learning model, and uploads the trained model to S3.
        """
        print('Start of loading train/test data...')
        # Load and process training data
        df_train = download_parquet_from_s3(S3_BUCKET_NAME, path(S3_DIR, TRAIN_PARQUET), path(TMP_DIR, TRAIN_PARQUET))
        df_train = process_na(df_train)
        df_train = df_train[df_train.fetch_date >= START_TRAIN_DATE]

        # Load and process test data
        df_test = download_parquet_from_s3(S3_BUCKET_NAME, path(S3_DIR, TEST_PARQUET), path(TMP_DIR, TEST_PARQUET))
        df_test = process_na(df_test)

        targets = df_train.attrs['target__']
        
        # Split features and targets for training and testing
        X_train, y_train = df_train.drop(columns=targets), df_train[targets]
        X_test, y_test = df_test.drop(columns=targets), df_test[targets]
        
        print("Data splitted:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Filter out columns with non-constant values in y_train
        target_cols = [col for col in y_train.columns if y_train[col].nunique() > 1]

        print('Start of model fitting...')
        model.fit(X_train, y_train[target_cols])

        # Save and upload the model to S3
        params = model.get_params()
        print("Model parameters:", params)

        s3_model_path = path(S3_DIR, FITTED_MODEL)
        local_model_path = path(TMP_DIR, FITTED_MODEL)
        joblib.dump(model, local_model_path)

        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        try:
            s3_hook.load_file(
                filename=local_model_path,
                key=s3_model_path,
                bucket_name=S3_BUCKET_NAME,
                replace=True
            )
            print(f"Model uploaded successfully to s3://{S3_BUCKET_NAME}/{s3_model_path}")
        except Exception as e:
            print(f"Error uploading model to S3: {e}")
        
        print('===Airflow finished all tasks.===')

    # Define Airflow tasks
    extract_task = PythonOperator(task_id='download_unzip_upload_id', python_callable=download_unzip_upload)    
    csv_to_parquet_task = PythonOperator(task_id='csv-extract-s3-parquet_id', python_callable=csv_extract)
    create_target_and_split_task = PythonOperator(task_id='create_target_and_split_id', python_callable=create_target_and_split)
    transform_fit_task = PythonOperator(task_id='transform_fit_task_id', python_callable=transform_fit)

    # Define task dependencies
    extract_task >> csv_to_parquet_task >> create_target_and_split_task >> transform_fit_task
