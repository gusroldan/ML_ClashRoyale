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
    bash_command=f'''
        set -e
        cd {PROJECT_ROOT} || {{ echo "Error: No se pudo cambiar a {PROJECT_ROOT}"; exit 1; }}
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado en {PROJECT_ROOT}"; ls -la; exit 1; }}
        echo "Verificando instalación de Kedro..."
        python -m kedro --version || {{ echo "Error: Kedro no está instalado"; exit 1; }}
        echo "Verificando que es un proyecto Kedro..."
        python -m kedro info || {{ echo "Error: No es un proyecto Kedro válido"; exit 1; }}
        echo "Ejecutando pipeline: data_preparation"
        python -m kedro run --pipeline=data_preparation
    ''',
    dag=dag,
)


