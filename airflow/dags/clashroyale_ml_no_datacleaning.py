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

regression = BashOperator(
    task_id='regression_pipeline',
    bash_command=f'''
        set -e
        cd {PROJECT_ROOT} || {{ echo "Error: No se pudo cambiar a {PROJECT_ROOT}"; exit 1; }}
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado en {PROJECT_ROOT}"; ls -la; exit 1; }}
        echo "Ejecutando pipeline: regression"
        python -m kedro run --pipeline=regression
    ''',
    dag=dag,
)

clustering = BashOperator(
    task_id='clustering_pipeline',
    bash_command=f'''
        set -e
        cd {PROJECT_ROOT} || {{ echo "Error: No se pudo cambiar a {PROJECT_ROOT}"; exit 1; }}
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado"; exit 1; }}
        echo "Verificando que train_data existe..."
        test -f data/05_model_input/train_data.csv || {{ echo "Error: train_data.csv no encontrado"; exit 1; }}
        echo "Ejecutando pipeline: unsupervised_learning"
        python -m kedro run --pipeline=unsupervised_learning
    ''',
    dag=dag,
)

dimensionality_reduction = BashOperator(
    task_id='dimensionality_reduction_pipeline',
    bash_command=f'''
        set -e
        cd {PROJECT_ROOT} || {{ echo "Error: No se pudo cambiar a {PROJECT_ROOT}"; exit 1; }}
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado"; exit 1; }}
        echo "Verificando que train_data existe..."
        test -f data/05_model_input/train_data.csv || {{ echo "Error: train_data.csv no encontrado"; exit 1; }}
        echo "Ejecutando pipeline: dimensionality_reduction"
        python -m kedro run --pipeline=dimensionality_reduction
    ''',
    dag=dag,
)

feature_engineering >> [classification, regression, clustering, dimensionality_reduction]


