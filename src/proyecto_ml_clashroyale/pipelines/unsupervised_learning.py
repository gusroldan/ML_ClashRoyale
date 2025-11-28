"""Pipeline para clustering (aprendizaje no supervisado)."""
## Objetivo: identificar patrones y grupos en los datos sin etiquetas.
from kedro.pipeline import Pipeline, node
from typing import Dict
from proyecto_ml_clashroyale.pipelines.nodes import unsupervised_learning_nodes


def create_unsupervised_learning_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de clustering.
    
    Returns:
        Pipeline de clustering
    """
    
    return Pipeline(
        [
            node(
                func=unsupervised_learning_nodes.train_kmeans,
                inputs=["train_data", "params:unsupervised_learning"],
                outputs=["kmeans_result", "kmeans_metrics"],
                name="train_kmeans_node",
                tags=["unsupervised_learning", "clustering", "kmeans"],
            ),
            node(
                func=unsupervised_learning_nodes.train_optics,
                inputs=["train_data", "params:unsupervised_learning"],
                outputs=["optics_result", "optics_metrics"],
                name="train_optics_node",
                tags=["unsupervised_learning", "clustering", "optics"],
            ),
            node(
                func=unsupervised_learning_nodes.train_hierarchical_clustering,
                inputs=["train_data", "params:unsupervised_learning"],
                outputs=["hierarchical_result", "hierarchical_metrics"],
                name="train_hierarchical_clustering_node",
                tags=["unsupervised_learning", "clustering", "hierarchical"],
            ),
            node(
                func=unsupervised_learning_nodes.create_clustering_comparison,
                inputs=["kmeans_result", "optics_result", "hierarchical_result"],
                outputs="clustering_comparison",
                name="create_clustering_comparison_node",
                tags=["unsupervised_learning", "evaluation"],
            ),
        ]
    )

