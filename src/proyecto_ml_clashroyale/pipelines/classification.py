"""Pipeline para clasificación."""
## Objetivo: predecir si el jugador A gana o pierde.
from kedro.pipeline import Pipeline, node
from typing import Dict
from proyecto_ml_clashroyale.pipelines.nodes import classification_nodes


def create_classification_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de clasificación.
    
    Returns:
        Pipeline de clasificación
    """
    
    return Pipeline(
        [
            node(
                func=classification_nodes.train_logistic_regression,
                inputs=["train_data", "test_data", "params:classification"],
                outputs=["logistic_result", "logistic_metrics"],
                name="train_logistic_regression_node",
                tags=["classification", "training", "logistic"],
            ),
            node(
                func=classification_nodes.train_random_forest,
                inputs=["train_data", "test_data", "params:classification"],
                outputs=["rf_clf_result", "rf_clf_metrics"],
                name="train_random_forest_node",
                tags=["classification", "training", "random_forest"],
            ),
            node(
                func=classification_nodes.train_xgboost,
                inputs=["train_data", "test_data", "params:classification"],
                outputs=["xgb_clf_result", "xgb_clf_metrics"],
                name="train_xgboost_node",
                tags=["classification", "training", "xgboost"],
            ),
            node(
                func=classification_nodes.train_svc,
                inputs=["train_data", "test_data", "params:classification"],
                outputs=["svc_result", "svc_metrics"],
                name="train_svc_node",
                tags=["classification", "training", "svc"],
            ),
            node(
                func=classification_nodes.train_lightgbm,
                inputs=["train_data", "test_data", "params:classification"],
                outputs=["lgbm_clf_result", "lgbm_clf_metrics"],
                name="train_lightgbm_node",
                tags=["classification", "training", "lightgbm"],
            ),
            node(
                func=classification_nodes.create_classification_comparison,
                inputs=["logistic_result", "rf_clf_result", "xgb_clf_result", "svc_result", "lgbm_clf_result"],
                outputs="classification_comparison",
                name="create_classification_comparison_node",
                tags=["classification", "evaluation"],
            ),
            node(
                func=classification_nodes.consolidate_classification_metrics,
                inputs="classification_comparison",
                outputs="classification_metrics",
                name="consolidate_classification_metrics_node",
                tags=["classification", "metrics"],
            ),
        ]
    )
