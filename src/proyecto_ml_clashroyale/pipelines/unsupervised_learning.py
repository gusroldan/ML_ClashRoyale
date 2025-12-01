"""Pipeline para clustering (aprendizaje no supervisado)."""
## Objetivo: identificar patrones y grupos en los datos sin etiquetas.
from kedro.pipeline import Pipeline, node
from typing import Dict
from proyecto_ml_clashroyale.pipelines.nodes import unsupervised_learning_nodes
from proyecto_ml_clashroyale.pipelines.nodes import cluster_analysis_nodes


def create_unsupervised_learning_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de clustering con análisis profundo.
    
    Returns:
        Pipeline de clustering con análisis de patrones
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
            node(
                func=unsupervised_learning_nodes.generate_clustering_visualizations,
                inputs=["train_data", "kmeans_result", "optics_result", "hierarchical_result", "params:unsupervised_learning"],
                outputs="clustering_visualizations",
                name="generate_clustering_visualizations_node",
                tags=["unsupervised_learning", "visualization"],
            ),
            # Análisis profundo de clusters - OPTICS (mejor modelo)
            node(
                func=cluster_analysis_nodes.analyze_cluster_statistics,
                inputs=["train_data", "optics_result", "params:unsupervised_learning"],
                outputs="optics_cluster_statistics",
                name="analyze_optics_cluster_statistics_node",
                tags=["unsupervised_learning", "cluster_analysis", "statistics"],
            ),
            node(
                func=cluster_analysis_nodes.create_cluster_profiles,
                inputs=["optics_cluster_statistics", "card_master_list"],
                outputs="optics_cluster_profiles",
                name="create_optics_cluster_profiles_node",
                tags=["unsupervised_learning", "cluster_analysis", "profiles"],
            ),
            node(
                func=cluster_analysis_nodes.create_business_interpretation,
                inputs="optics_cluster_profiles",
                outputs="optics_business_interpretation",
                name="create_optics_business_interpretation_node",
                tags=["unsupervised_learning", "cluster_analysis", "business"],
            ),
        ]
    )

