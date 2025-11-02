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
    'classification_ml_pipeline',
    default_args=default_args,
    description='Feature Engineering + Classification (only)',
    schedule=None,
    catchup=False,
)

# Path del proyecto en el contenedor Docker
PROJECT_ROOT = "/app"

feature_engineering = BashOperator(
    task_id='feature_engineering',
    bash_command=f'''
        set -e
        cd {PROJECT_ROOT} || {{ echo "Error: No se pudo cambiar a {PROJECT_ROOT}"; exit 1; }}
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado en {PROJECT_ROOT}"; ls -la; exit 1; }}
        echo "Ejecutando pipeline: feature_engineering"
        python -m kedro run --pipeline=feature_engineering
    ''',
    dag=dag,
)

classification = BashOperator(
    task_id='classification_pipeline',
    bash_command=f'''
        set -e
        cd {PROJECT_ROOT} || {{ echo "Error: No se pudo cambiar a {PROJECT_ROOT}"; exit 1; }}
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado en {PROJECT_ROOT}"; ls -la; exit 1; }}
        echo "Ejecutando pipeline: classification"
        python -m kedro run --pipeline=classification
    ''',
    dag=dag,
)

feature_engineering >> classification


