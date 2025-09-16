"""Pipeline para la Fase 1: Comprensión del Negocio (CRISP-DM)."""

from kedro.pipeline import Pipeline, node
from .nodes import business_understanding_nodes


def create_business_understanding_pipeline() -> Pipeline:
    """Crear pipeline para la comprensión del negocio y limpieza inicial.
    
    Este pipeline implementa:
    - Limpieza inicial de datos (renombrado de primera columna)
    - Definir objetivos del proyecto
    - Evaluar la situación actual
    - Determinar objetivos de Machine Learning
    - Producir plan del proyecto
    
    Returns:
        Pipeline configurado para la comprensión del negocio y limpieza
    """
    return Pipeline(
        [
            # Limpieza inicial de datos - renombrar primera columna
            node(
                func=business_understanding_nodes.clean_and_rename_first_column,
                inputs=["Combates1"],
                outputs="Combates1_cleaned",
                name="clean_combates1",
                tags=["business_understanding", "data_cleaning"]
            ),
            node(
                func=business_understanding_nodes.clean_and_rename_first_column,
                inputs=["Combates2"],
                outputs="Combates2_cleaned",
                name="clean_combates2",
                tags=["business_understanding", "data_cleaning"]
            ),
            node(
                func=business_understanding_nodes.clean_and_rename_first_column,
                inputs=["Combates3"],
                outputs="Combates3_cleaned",
                name="clean_combates3",
                tags=["business_understanding", "data_cleaning"]
            ),
            # Análisis de objetivos del negocio
            node(
                func=business_understanding_nodes.analyze_business_objectives,
                inputs=["Combates1_cleaned", "Combates2_cleaned", "Combates3_cleaned", "card_master_list", "wincons"],
                outputs="business_objectives_analysis",
                name="analyze_business_objectives",
                tags=["business_understanding", "objectives"]
            ),
            # Evaluación de situación actual
            node(
                func=business_understanding_nodes.evaluate_current_situation,
                inputs=["Combates1_cleaned", "Combates2_cleaned", "Combates3_cleaned"],
                outputs="current_situation_evaluation",
                name="evaluate_current_situation",
                tags=["business_understanding", "situation"]
            ),
            # Definición de objetivos de ML
            node(
                func=business_understanding_nodes.define_ml_objectives,
                inputs=["business_objectives_analysis", "current_situation_evaluation"],
                outputs="ml_objectives_definition",
                name="define_ml_objectives",
                tags=["business_understanding", "ml_objectives"]
            ),
            # Generación del plan del proyecto
            node(
                func=business_understanding_nodes.generate_project_plan,
                inputs=["business_objectives_analysis", "current_situation_evaluation", "ml_objectives_definition"],
                outputs="project_plan",
                name="generate_project_plan",
                tags=["business_understanding", "project_plan"]
            ),
            # Creación del resumen ejecutivo
            node(
                func=business_understanding_nodes.create_business_summary,
                inputs=["business_objectives_analysis", "current_situation_evaluation", 
                       "ml_objectives_definition", "project_plan"],
                outputs="business_understanding_summary",
                name="create_business_summary",
                tags=["business_understanding", "summary"]
            )
        ],
        tags="business_understanding"
    )
