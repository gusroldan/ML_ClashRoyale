from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import os
import logging

# Configuración parametrizable desde variables de entorno
PROJECT_ROOT = os.getenv('KEDRO_PROJECT_ROOT', '/app')
KEDRO_ENV = os.getenv('KEDRO_ENV', 'base')
MAX_RETRIES = int(os.getenv('AIRFLOW_MAX_RETRIES', '1'))
RETRY_DELAY = int(os.getenv('AIRFLOW_RETRY_DELAY_MINUTES', '5'))

default_args = {
    'owner': 'clashroyale_ml',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 26),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': MAX_RETRIES,
    'retry_delay': timedelta(minutes=RETRY_DELAY),
    'on_failure_callback': None,  # Puede configurarse para notificaciones
    'on_retry_callback': None,
    'on_success_callback': None,
}

dag = DAG(
    'clashroyale_ml_with_datacleaning',
    default_args=default_args,
    description='Pipeline ML completo: data_engineering → supervised → unsupervised para Clash Royale',
    schedule='@weekly',
    catchup=False,
    tags=['ml', 'clashroyale', 'kedro', 'complete_pipeline'],
    params={
        'project_root': PROJECT_ROOT,
        'kedro_env': KEDRO_ENV,
        'max_samples_per_class': 5000,  # Parametrizable
    },
    doc_md="""
    # Pipeline ML Completo - Clash Royale
    
    Este DAG ejecuta el pipeline completo de Machine Learning:
    
    1. **Business Understanding**: Limpieza inicial y comprensión del negocio
    2. **Data Cleaning**: Preparación y combinación de datasets
    3. **Feature Engineering**: Creación de features para ML
    4. **Supervised Learning**: Clasificación y Regresión
    5. **Unsupervised Learning**: Clustering, Reducción Dimensional, Detección de Anomalías, Reglas de Asociación
    
    ## Parámetros Configurables
    
    - `KEDRO_PROJECT_ROOT`: Directorio raíz del proyecto (default: `/app`)
    - `KEDRO_ENV`: Entorno de Kedro (default: `base`)
    - `AIRFLOW_MAX_RETRIES`: Número máximo de reintentos (default: `1`)
    - `AIRFLOW_RETRY_DELAY_MINUTES`: Delay entre reintentos en minutos (default: `5`)
    
    ## Logs
    
    Los logs de cada tarea están disponibles en la interfaz de Airflow.
    Los logs de Kedro se guardan en `data/logs/` dentro del proyecto.
    """
)

# Función helper para comandos Bash con mejor manejo de errores
def create_kedro_command(pipeline_name: str, additional_checks: list = None) -> str:
    """Crear comando Bash para ejecutar pipeline de Kedro con manejo de errores robusto.
    
    Args:
        pipeline_name: Nombre del pipeline a ejecutar
        additional_checks: Lista de archivos/directorios a verificar antes de ejecutar
        
    Returns:
        Comando Bash completo
    """
    checks = additional_checks or []
    check_commands = "\n".join([
        f'        test -f {check} || {{ echo "Error: {check} no encontrado"; exit 1; }}'
        for check in checks
    ])
    
    return f'''
        set -e
        set -o pipefail
        cd {{{{ params.project_root }}}} || {{ echo "Error: No se pudo cambiar a directorio del proyecto"; exit 1; }}
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Iniciando pipeline: {pipeline_name}"
        echo "Directorio actual: $(pwd)"
        echo "Verificando que pyproject.toml existe..."
        test -f pyproject.toml || {{ echo "Error: pyproject.toml no encontrado"; ls -la; exit 1; }}
{check_commands}
        echo "Ejecutando pipeline: {pipeline_name}"
        python -m kedro run --pipeline={pipeline_name} --env={{{{ params.kedro_env }}}} 2>&1 | tee -a data/logs/airflow_{pipeline_name}.log
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Pipeline {pipeline_name} completado exitosamente"
        else
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Pipeline {pipeline_name} falló con código $EXIT_CODE"
            exit $EXIT_CODE
        fi
    '''

# Tarea 1: Business Understanding (genera Combates1_cleaned, Combates2_cleaned, Combates3_cleaned)
business_understanding = BashOperator(
    task_id='business_understanding',
    bash_command=create_kedro_command(
        'business_understanding',
        ['data/01_raw/battlesStaging_12272020_WL_tagged.csv']
    ),
    dag=dag,
)

# Tarea 2: Preparación de datos (necesita Combates1_cleaned, Combates2_cleaned, Combates3_cleaned)
data_cleaning = BashOperator(
    task_id='data_cleaning',
    bash_command=create_kedro_command(
        'data_preparation',
        ['data/02_intermediate/Combates1_cleaned.csv']
    ),
    dag=dag,
)

# Tarea 3: Feature Engineering
feature_engineering = BashOperator(
    task_id='feature_engineering',
    bash_command=create_kedro_command('feature_engineering'),
    dag=dag,
)

# Tarea 4: Pipeline de Clasificación
classification = BashOperator(
    task_id='classification_pipeline',
    bash_command=create_kedro_command('classification', ['data/05_model_input/train_data.csv']),
    dag=dag,
)

# Tarea 5: Pipeline de Regresión
regression = BashOperator(
    task_id='regression_pipeline',
    bash_command=create_kedro_command('regression', ['data/05_model_input/train_data.csv']),
    dag=dag,
)

# Tarea 6: Pipeline de Clustering (Aprendizaje No Supervisado)
clustering = BashOperator(
    task_id='clustering_pipeline',
    bash_command=create_kedro_command('unsupervised_learning', ['data/05_model_input/train_data.csv']),
    dag=dag,
)

# Tarea 7: Pipeline de Reducción de Dimensionalidad
dimensionality_reduction = BashOperator(
    task_id='dimensionality_reduction_pipeline',
    bash_command=create_kedro_command('dimensionality_reduction', ['data/05_model_input/train_data.csv']),
    dag=dag,
)

# Tarea 8: Pipeline de Detección de Anomalías (Aprendizaje No Supervisado)
anomaly_detection = BashOperator(
    task_id='anomaly_detection_pipeline',
    bash_command=create_kedro_command('anomaly_detection', ['data/05_model_input/train_data.csv']),
    dag=dag,
)

# Tarea 9: Pipeline de Reglas de Asociación (Aprendizaje No Supervisado)
association_rules = BashOperator(
    task_id='association_rules_pipeline',
    bash_command=create_kedro_command('association_rules', ['data/05_model_input/train_data.csv']),
    dag=dag,
)

# Definir orden de ejecución
# Flujo: data_engineering → supervised → unsupervised
data_engineering = [business_understanding, data_cleaning, feature_engineering]
supervised = [classification, regression]
unsupervised = [clustering, dimensionality_reduction, anomaly_detection, association_rules]

# Dependencias: data_engineering → supervised → unsupervised
business_understanding >> data_cleaning >> feature_engineering
feature_engineering >> supervised
feature_engineering >> unsupervised


