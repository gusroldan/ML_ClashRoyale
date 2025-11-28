"""Pipeline para reducción de dimensionalidad."""
## Objetivo: reducir la dimensionalidad de los datos manteniendo la información más importante.
from kedro.pipeline import Pipeline, node
from typing import Dict
from proyecto_ml_clashroyale.pipelines.nodes import dimensionality_reduction_nodes


def create_dimensionality_reduction_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de reducción de dimensionalidad.
    
    Returns:
        Pipeline de reducción de dimensionalidad
    """
    
    return Pipeline(
        [
            node(
                func=dimensionality_reduction_nodes.apply_pca,
                inputs=["train_data", "params:dimensionality_reduction"],
                outputs=["pca_result", "pca_metrics"],
                name="apply_pca_node",
                tags=["dimensionality_reduction", "pca"],
            ),
            node(
                func=dimensionality_reduction_nodes.apply_umap,
                inputs=["train_data", "params:dimensionality_reduction"],
                outputs=["umap_result", "umap_metrics"],
                name="apply_umap_node",
                tags=["dimensionality_reduction", "umap"],
            ),
            node(
                func=dimensionality_reduction_nodes.create_dimensionality_reduction_comparison,
                inputs=["pca_result", "umap_result"],
                outputs="dimensionality_reduction_comparison",
                name="create_dimensionality_reduction_comparison_node",
                tags=["dimensionality_reduction", "evaluation"],
            ),
        ]
    )

