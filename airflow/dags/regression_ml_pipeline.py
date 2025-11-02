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
    'regression_ml_pipeline',
    default_args=default_args,
    description='Feature Engineering + Regression (only)',
    schedule=None,
    catchup=False,
)

# Path del proyecto en el contenedor Docker
PROJECT_ROOT = "/app"

feature_engineering = BashOperator(
    task_id='feature_engineering',
    bash_command=f'cd {PROJECT_ROOT} && set -e && python -m kedro run --pipeline=feature_engineering',
    dag=dag,
)

regression = BashOperator(
    task_id='regression_pipeline',
    bash_command=f'cd {PROJECT_ROOT} && set -e && python -m kedro run --pipeline=regression',
    dag=dag,
)

feature_engineering >> regression


