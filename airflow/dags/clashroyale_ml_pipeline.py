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
    'clashroyale_ml_with_datacleaning',
    default_args=default_args,
    description='Pipeline ML completo (incluye data_preparation) para Clash Royale',
    schedule='@weekly',
    catchup=False,
)

# Tarea 1: Preparación de datos (pipelines existentes)
data_cleaning = BashOperator(
    task_id='data_cleaning',
    bash_command='cd /mnt/c/Users/Usuario/Documents/GitHub/ML_ClashRoyale && set -e && venv/bin/python -m kedro run --pipeline=data_preparation',
    dag=dag,
)

# Tarea 2: Feature Engineering
feature_engineering = BashOperator(
    task_id='feature_engineering',
    bash_command='cd /mnt/c/Users/Usuario/Documents/GitHub/ML_ClashRoyale && set -e && venv/bin/python -m kedro run --pipeline=feature_engineering',
    dag=dag,
)

# Tarea 3: Pipeline de Clasificación
classification = BashOperator(
    task_id='classification_pipeline',
    bash_command='cd /mnt/c/Users/Usuario/Documents/GitHub/ML_ClashRoyale && set -e && venv/bin/python -m kedro run --pipeline=classification',
    dag=dag,
)

# Tarea 4: Pipeline de Regresión
regression = BashOperator(
    task_id='regression_pipeline',
    bash_command='cd /mnt/c/Users/Usuario/Documents/GitHub/ML_ClashRoyale && set -e && venv/bin/python -m kedro run --pipeline=regression',
    dag=dag,
)

# Definir orden de ejecución
data_cleaning >> feature_engineering >> [classification, regression]


