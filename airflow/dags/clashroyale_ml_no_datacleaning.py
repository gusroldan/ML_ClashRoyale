from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'clashroyale_ml',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 26),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'clashroyale_ml_no_datacleaning',
    default_args=default_args,
    description='Pipeline ML sin data_preparation: feature_engineering -> classification & regression',
    schedule=None,
    catchup=False,
)

PROJECT_ROOT = "/mnt/c/Users/Usuario/Documents/GitHub/ML_ClashRoyale"

feature_engineering = BashOperator(
    task_id='feature_engineering',
    bash_command=f'cd {PROJECT_ROOT} && set -e && source venv/bin/activate && kedro run --pipeline=feature_engineering',
    dag=dag,
)

classification = BashOperator(
    task_id='classification_pipeline',
    bash_command=f'cd {PROJECT_ROOT} && set -e && source venv/bin/activate && kedro run --pipeline=classification',
    dag=dag,
)

regression = BashOperator(
    task_id='regression_pipeline',
    bash_command=f'cd {PROJECT_ROOT} && set -e && source venv/bin/activate && kedro run --pipeline=regression',
    dag=dag,
)

feature_engineering >> [classification, regression]


