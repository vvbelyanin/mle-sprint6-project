from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import os
import subprocess

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG(
    'download_and_upload_to_s3_ver2',
    default_args=default_args,
    schedule=None,
    catchup=False
) as dag:
    
    download_path = '/tmp/data.zip'
    csv_output_path = '/tmp/data.csv'
    s3_bucket_name = os.getenv('S3_BUCKET_NAME')
    s3_key = 'recsys/sprint_6_data/data.csv'
    
    def download_unzip_upload():
        download_command = f'wget -O {download_path} $(yadisk-direct https://disk.yandex.com/d/Io0siOESo2RAaA)'
        subprocess.run(download_command, shell=True, check=True)
        print(f"Downloaded file to {download_path}")
        
        unzip_command = f'unzip -p {download_path} train_ver2.csv > {csv_output_path}'
        subprocess.run(unzip_command, shell=True, check=True)
        print(f"Unzipped file to {csv_output_path}")
        
        s3_hook = S3Hook(aws_conn_id='yandexcloud_default')
        s3_hook.load_file(
            filename=csv_output_path,
            key=s3_key,
            bucket_name=s3_bucket_name,
            replace=True
        )
        print(f"Uploaded file to S3: {s3_bucket_name}/{s3_key}")
    
    extract_task = PythonOperator(
        task_id='download_unzip_upload',
        python_callable=download_unzip_upload,
    )

