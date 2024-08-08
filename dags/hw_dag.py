from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'hw_dag',
    default_args=default_args,
    description='A simple ML pipeline DAG',
    schedule_interval=timedelta(days=1),
)

def run_pipeline():
    os.system('python3 ~/airflow_hw/modules/pipeline.py')

def run_predict():
    os.system('python3 ~/airflow_hw/modules/predict.py')

pipeline_task = PythonOperator(
    task_id='pipeline',
    python_callable=run_pipeline,
    dag=dag,
)

predict_task = PythonOperator(
    task_id='predict',
    python_callable=run_predict,
    dag=dag,
)

pipeline_task >> predict_task
