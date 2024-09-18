from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import pandas as pd
import os
import subprocess
import joblib

from data_utils import (get_mem_usage, new_columns, attrs, DataFrameProcessor, 
                        frequency_encoding, get_X_y, gen_random_data, show_s3_folder,
                        process_na, process_df, model)

default_args = {
    'start_date': datetime(2023, 9, 17),
}

dtype_spec: dict = {
    'age': 'str',
    'tenure_months': 'str',
    'client_type_1m': 'str',
    'ind_spouse_employee': 'str',
}

s3_bucket_name = os.getenv('S3_BUCKET_NAME')
s3_key = 'recsys/sprint_6_data/data.csv'
s3_key_parquet = 'recsys/sprint_6_data/data.parquet'
local_parquet_file = '/tmp/data.parquet'
download_path = '/tmp/data.zip'
csv_output_path = '/tmp/data.csv'
aws_conn_id_default = 'yandexcloud_default'

def download_parquet_from_s3(bucket_name, s3_key, local_path, aws_conn_id=aws_conn_id_default):
    s3_hook = S3Hook(aws_conn_id=aws_conn_id)
    s3_hook.get_key(key=s3_key, bucket_name=bucket_name).download_file(local_path)
    df = pd.read_parquet(local_path)
    os.remove(local_path)
    return df

def upload_dataframe_to_s3_as_parquet(df, bucket_name, s3_key, local_path, aws_conn_id=aws_conn_id_default):
    df.to_parquet(local_path)
    s3_hook = S3Hook(aws_conn_id=aws_conn_id)
    s3_hook.load_file(filename=local_path, key=s3_key, bucket_name=bucket_name, replace=True)
    os.remove(local_path)

with DAG(
    dag_id='extract_yandexcloud_to_s3',
    default_args=default_args,
    schedule=None,
    catchup=False
) as dag:
        
    def download_unzip_upload():
        download_command = f'wget -O {download_path} $(yadisk-direct https://disk.yandex.com/d/Io0siOESo2RAaA)'
        subprocess.run(download_command, shell=True, check=True)
        
        unzip_command = f'unzip -p {download_path} train_ver2.csv > {csv_output_path}'
        subprocess.run(unzip_command, shell=True, check=True)
        
        s3_hook = S3Hook(aws_conn_id=aws_conn_id_default)
        s3_hook.load_file(filename=csv_output_path, key=s3_key, bucket_name=s3_bucket_name, replace=True)
    
    def csv_extract():
        s3_hook = S3Hook(aws_conn_id=aws_conn_id_default)
        s3_object = s3_hook.get_key(key=s3_key, bucket_name=s3_bucket_name)
        s3_body = s3_object.get()['Body']
        df = pd.read_csv(
            s3_body, 
            dtype=dtype_spec,
            parse_dates=['fetch_date', 'registration_date', 'last_date_as_primary'],
            date_format='%Y-%m-%d',
            low_memory=False,
            header=0,
            names=list(new_columns.values())
        )
        df.attrs = attrs
        df.to_parquet(local_parquet_file)
        s3_hook.load_file(filename=local_parquet_file, key=s3_key_parquet, bucket_name=s3_bucket_name, replace=True)
        get_mem_usage()


    def create_target_and_split():
        df = download_parquet_from_s3(s3_bucket_name,'recsys/sprint_6_data/data.parquet', '/tmp/data.parquet')

        product_features = [col for col in df.columns if col.startswith('ind_1m_')]
        grouped = df.groupby('id')[product_features]

        shifted_values = grouped.shift(-1)
        df_targets = ((df[product_features] == 0) & (shifted_values == 1)).astype(int).add_prefix('target__')
        for col in df_targets.columns:
            df[col] = df_targets[col]
        del shifted_values, df_targets

        targets = [col for col in df.columns if col.startswith("target__")]
        last_row_mask = df.groupby('id')['id'].transform('shift', -1).notna()
        df = df[last_row_mask]

        grouped_0 = (df.groupby('id')[targets].sum().sum(axis=1) == 0)
        df = df[~df.id.isin(grouped_0[grouped_0].index)]
        df.attrs['target__'] = targets
        get_mem_usage()

        split_date = '2016-02-28'

        df_test_local = '/tmp/data_test.parquet'
        upload_dataframe_to_s3_as_parquet(df[df.fetch_date > split_date], s3_bucket_name, 'recsys/sprint_6_data/df_test.parquet', df_test_local)

        df_train_local = '/tmp/data_train.parquet'
        df = df[df.fetch_date <= split_date]
        upload_dataframe_to_s3_as_parquet(df, s3_bucket_name, 'recsys/sprint_6_data/df_train.parquet', df_train_local)


    def transform_fit():
        start_date_of_train = '2015-07-28'

        df_train = download_parquet_from_s3(s3_bucket_name,'recsys/sprint_6_data/df_train.parquet', '/tmp/df_train.parquet')
        df_train = process_na(df_train)
        df_train = df_train[df_train.fetch_date >= start_date_of_train]
        print('Train data loaded.')

        df_test = download_parquet_from_s3(s3_bucket_name,'recsys/sprint_6_data/df_test.parquet', '/tmp/df_test.parquet')
        df_test = process_na(df_test)
        print('Test data loaded.')

        targets = df_train.attrs['target__']
        
        X_train, y_train = df_train.drop(columns=targets), df_train[targets]
        X_test, y_test = df_test.drop(columns=targets), df_test[targets]

        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}m y_test: {y_test.shape}")

        target_cols = [col for col in y_train.columns if y_train[col].nunique() > 1]

        print('Start of model fitting')
        model.fit(X_train, y_train[target_cols])

        params = model.get_params()
        print("Model parameters:", params)

        s3_file_path = 'recsys/sprint_6_data/fitted_model.joblib'
        local_model_path = '/tmp/fitted_model.joblib'
        joblib.dump(model, local_model_path)

        s3_hook = S3Hook(aws_conn_id=aws_conn_id_default)
        try:
            # Upload the file to the specified S3 bucket and path
            s3_hook.load_file(
                filename=local_model_path,
                key=s3_file_path,
                bucket_name=s3_bucket_name,
                replace=True
            )
            print(f"Model uploaded successfully to s3://{s3_bucket_name}/{s3_file_path}")
        except Exception as e:
            print(f"Error uploading model to S3: {e}")

#  extract_task = EmptyOperator(task_id='extract_task')

    extract_task = PythonOperator(task_id='download_unzip_upload_id', python_callable=download_unzip_upload)    
    csv_to_parquet_task = PythonOperator(task_id='csv-extract-s3-parquet_id', python_callable=csv_extract)
    create_target_and_split_task = PythonOperator(task_id='create_target_and_split_id', python_callable=create_target_and_split)
    transform_fit_task = PythonOperator(task_id='transform_fit_task_id', python_callable=transform_fit)

    extract_task >> csv_to_parquet_task >> create_target_and_split_task >> transform_fit_task