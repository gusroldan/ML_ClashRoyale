"""Pipeline para regresión."""
##Objetivo: predecir el cambio de trofeos del jugador A.

from kedro.pipeline import Pipeline, node
from typing import Dict
from proyecto_ml_clashroyale.pipelines.nodes import regression_nodes


def create_regression_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de regresión.
    
    Returns:
        Pipeline de regresión
    """
    
    return Pipeline(
        [
            node(
                func=regression_nodes.train_linear_regression,
                inputs=["train_data", "test_data", "params:regression"],
                outputs=["linear_result", "linear_metrics"],
                name="train_linear_regression_node",
                tags=["regression", "training", "linear"],
            ),
            node(
                func=regression_nodes.train_ridge,
                inputs=["train_data", "test_data", "params:regression"],
                outputs=["ridge_result", "ridge_metrics"],
                name="train_ridge_node",
                tags=["regression", "training", "ridge"],
            ),
            node(
                func=regression_nodes.train_random_forest,
                inputs=["train_data", "test_data", "params:regression"],
                outputs=["rf_reg_result", "rf_reg_metrics"],
                name="train_random_forest_regressor_node",
                tags=["regression", "training", "random_forest"],
            ),
            node(
                func=regression_nodes.train_xgboost,
                inputs=["train_data", "test_data", "params:regression"],
                outputs=["xgb_reg_result", "xgb_reg_metrics"],
                name="train_xgboost_regressor_node",
                tags=["regression", "training", "xgboost"],
            ),
            node(
                func=regression_nodes.train_svr,
                inputs=["train_data", "test_data", "params:regression"],
                outputs=["svr_result", "svr_metrics"],
                name="train_svr_node",
                tags=["regression", "training", "svr"],
            ),
            node(
                func=regression_nodes.train_lightgbm,
                inputs=["train_data", "test_data", "params:regression"],
                outputs=["lgbm_reg_result", "lgbm_reg_metrics"],
                name="train_lightgbm_regressor_node",
                tags=["regression", "training", "lightgbm"],
            ),
            node(
                func=regression_nodes.create_regression_comparison,
                inputs=["linear_result", "ridge_result", "rf_reg_result", "xgb_reg_result", "svr_result", "lgbm_reg_result"],
                outputs="regression_comparison",
                name="create_regression_comparison_node",
                tags=["regression", "evaluation"],
            ),
            node(
                func=regression_nodes.consolidate_regression_metrics,
                inputs="regression_comparison",
                outputs="regression_metrics",
                name="consolidate_regression_metrics_node",
                tags=["regression", "metrics"],
            ),
        ]
    )
