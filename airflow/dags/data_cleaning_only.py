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
    'data_cleaning_only',
    default_args=default_args,
    description='Solo ejecución de data preparation/cleaning',
    schedule=None,
    catchup=False,
)

# Path del proyecto en el contenedor Docker
PROJECT_ROOT = "/app"

data_cleaning = BashOperator(
    task_id='data_cleaning',
    bash_command=f'cd {PROJECT_ROOT} && set -e && python -m kedro run --pipeline=data_preparation',
    dag=dag,
)


