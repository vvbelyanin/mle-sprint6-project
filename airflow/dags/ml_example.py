from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import os

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

# Define the DAG
with DAG(
    'download_and_upload_to_s3',
    default_args=default_args,
    schedule=None,  # You can set this to a cron expression if you want it to run periodically
    catchup=False
) as dag:
    
    download_path = '/tmp/data.zip'
    csv_output_path = '/tmp/data.csv'
    s3_bucket_name = os.getenv('S3_BUCKET_NAME')
    s3_key = 'recsys/sprint_6_data/data.csv'
    
    download_file = BashOperator(
        task_id='download_file',
        bash_command=f'wget -O {download_path} $(yadisk-direct https://disk.yandex.com/d/Io0siOESo2RAaA)',
    )
    
    unzip_file = BashOperator(
        task_id='unzip_file',
        bash_command=f'unzip -p {download_path} train_ver2.csv > {csv_output_path}',
    )
    
    def upload_to_s3():
        s3_hook = S3Hook(aws_conn_id='yandexcloud_default')
        s3_hook.load_file(
            filename=csv_output_path,
            key=s3_key,
            bucket_name=s3_bucket_name,
            replace=True
        )
    
    upload_task = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3,
    )
    
    download_file >> unzip_file >> upload_task


from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

def extract():
    # Simulate data extraction
    data = {'age': [25, 30, 35, 40], 'salary': [50000, 60000, 70000, 80000], 'bought_insurance': [0, 1, 0, 1]}
    df = pd.DataFrame(data)
    
    # Save extracted data to a file
    df.to_csv('/tmp/extracted_data.csv', index=False)
    print("Data extracted and saved to /tmp/extracted_data.csv")

def transform():
    # Load the extracted data
    df = pd.read_csv('/tmp/extracted_data.csv')

    # Simulate data transformation
    df['salary'] = df['salary'] * 1.1  # Example transformation: salary increase
    df.to_csv('/tmp/transformed_data.csv', index=False)
    print("Data transformed and saved to /tmp/transformed_data.csv")

def load():
    # Load the transformed data
    df = pd.read_csv('/tmp/transformed_data.csv')

    # Simulate loading the data (e.g., save to a database or S3)
    df.to_csv('/tmp/loaded_data.csv', index=False)
    print("Data loaded and saved to /tmp/loaded_data.csv")

def model():
    df = pd.read_csv('/tmp/loaded_data.csv')

    # Features and target
    X = df[['age', 'salary']]
    y = df['bought_insurance']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the trained model to a file
    with open('/tmp/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved to /tmp/logistic_regression_model.pkl")

with DAG('etl_and_modeling_intermediate', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    extract_task = PythonOperator(task_id='extract', python_callable=extract)
    transform_task = PythonOperator(task_id='transform', python_callable=transform)
    load_task = PythonOperator(task_id='load', python_callable=load)
    model_task = PythonOperator(task_id='model', python_callable=model)

    # Define task dependencies
    extract_task >> transform_task >> load_task >> model_task
