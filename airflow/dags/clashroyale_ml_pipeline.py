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

# Path del proyecto en el contenedor Docker
PROJECT_ROOT = "/app"

# Tarea 1: Business Understanding (genera Combates1_cleaned, Combates2_cleaned, Combates3_cleaned)
business_understanding = BashOperator(
    task_id='business_understanding',
    bash_command=f'''
        set -e
        cd {PROJECT_ROOT} || {{ echo "Error: No se pudo cambiar a {PROJECT_ROOT}"; exit 1; }}
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado en {PROJECT_ROOT}"; ls -la; exit 1; }}
        echo "Verificando que los datos de entrada existen..."
        test -f data/01_raw/battlesStaging_12272020_WL_tagged.csv || {{ echo "Error: Datos de entrada no encontrados"; exit 1; }}
        echo "Ejecutando pipeline: business_understanding"
        python -m kedro run --pipeline=business_understanding
    ''',
    dag=dag,
)

# Tarea 2: Preparación de datos (necesita Combates1_cleaned, Combates2_cleaned, Combates3_cleaned)
data_cleaning = BashOperator(
    task_id='data_cleaning',
    bash_command=f'''
        set -e
        cd {PROJECT_ROOT} || {{ echo "Error: No se pudo cambiar a {PROJECT_ROOT}"; exit 1; }}
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado"; exit 1; }}
        echo "Verificando que los archivos cleaned existen..."
        test -f data/02_intermediate/Combates1_cleaned.csv || {{ echo "Error: Combates1_cleaned.csv no encontrado"; exit 1; }}
        echo "Ejecutando pipeline: data_preparation"
        python -m kedro run --pipeline=data_preparation
    ''',
    dag=dag,
)

# Tarea 3: Feature Engineering
feature_engineering = BashOperator(
    task_id='feature_engineering',
    bash_command=f'''
        set -e
        cd {PROJECT_ROOT} || {{ echo "Error: No se pudo cambiar a {PROJECT_ROOT}"; exit 1; }}
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado"; exit 1; }}
        echo "Ejecutando pipeline: feature_engineering"
        python -m kedro run --pipeline=feature_engineering
    ''',
    dag=dag,
)

# Tarea 4: Pipeline de Clasificación
classification = BashOperator(
    task_id='classification_pipeline',
    bash_command=f'''
        set -e
        cd {PROJECT_ROOT} || {{ echo "Error: No se pudo cambiar a {PROJECT_ROOT}"; exit 1; }}
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado"; exit 1; }}
        echo "Ejecutando pipeline: classification"
        python -m kedro run --pipeline=classification
    ''',
    dag=dag,
)

# Tarea 5: Pipeline de Regresión
regression = BashOperator(
    task_id='regression_pipeline',
    bash_command=f'''
        set -e
        cd {PROJECT_ROOT} || {{ echo "Error: No se pudo cambiar a {PROJECT_ROOT}"; exit 1; }}
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado"; exit 1; }}
        echo "Ejecutando pipeline: regression"
        python -m kedro run --pipeline=regression
    ''',
    dag=dag,
)

# Tarea 6: Pipeline de Clustering (Aprendizaje No Supervisado)
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

# Tarea 7: Pipeline de Reducción de Dimensionalidad
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

# Definir orden de ejecución
business_understanding >> data_cleaning >> feature_engineering >> [classification, regression, clustering, dimensionality_reduction]


