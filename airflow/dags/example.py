from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 9, 17),
    'retries': 1,
}

with DAG('etl_dag', default_args=default_args, description='A simple ETL DAG', schedule='@daily', catchup=False) as dag:

    extract_task = BashOperator(task_id='extract_data', bash_command='echo "Extracting data..."')

    def transform_data():
        transformed_data = "Transformed Data"
        return transformed_data

    transform_task = PythonOperator(task_id='transform_data', python_callable=transform_data)

    def load_data():
        print("Data successfully loaded!")

    load_task = PythonOperator(task_id='load_data', python_callable=load_data)

    extract_task >> transform_task >> load_task
