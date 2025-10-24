"""Pipeline para preparación de datos (Fase 3 CRISP-DM)."""

from kedro.pipeline import Pipeline, node

from .nodes import data_preparation_nodes


def create_data_preparation_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de preparación de datos.
    
    Returns:
        Pipeline de preparación de datos configurado
    """
    return Pipeline(
        [
            node(
                func=data_preparation_nodes.select_relevant_columns,
                inputs=[
                    "Combates1_cleaned",
                    "Combates2_cleaned",
                    "Combates3_cleaned"
                ],
                outputs="selected_datasets",
                name="select_relevant_columns",
                tags=["data_preparation", "column_selection"]
            ),
            node(
                func=data_preparation_nodes.combine_datasets,
                inputs=["selected_datasets"],
                outputs="combined_dataset",
                name="combine_datasets",
                tags=["data_preparation", "data_combination"]
            ),
            node(
                func=data_preparation_nodes.validate_combined_dataset,
                inputs=["combined_dataset"],
                outputs="dataset_validation",
                name="validate_combined_dataset",
                tags=["data_preparation", "validation"]
            ),
            node(
                func=data_preparation_nodes.create_preparation_summary,
                inputs=["dataset_validation"],
                outputs="data_preparation_summary",
                name="create_preparation_summary",
                tags=["data_preparation", "summary"]
            )
        ]
    )
