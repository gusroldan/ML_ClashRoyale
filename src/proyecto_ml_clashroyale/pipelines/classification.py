"""Pipeline para clasificación."""
## Objetivo: predecir si el jugador A gana o pierde.
from kedro.pipeline import Pipeline, node
from typing import Dict, Any
from proyecto_ml_clashroyale.pipelines.nodes import classification_nodes
from proyecto_ml_clashroyale.pipelines.nodes import shap_analysis_nodes


def _calculate_rf_shap(model_result: Dict[str, Any], train_data, test_data, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper para calcular SHAP de Random Forest."""
    params = parameters.copy()
    params['model_name'] = 'Random Forest'
    return shap_analysis_nodes.calculate_shap_values_classification(model_result, train_data, test_data, params)


def _calculate_xgb_shap(model_result: Dict[str, Any], train_data, test_data, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper para calcular SHAP de XGBoost."""
    params = parameters.copy()
    params['model_name'] = 'XGBoost'
    return shap_analysis_nodes.calculate_shap_values_classification(model_result, train_data, test_data, params)


def _calculate_lgbm_shap(model_result: Dict[str, Any], train_data, test_data, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper para calcular SHAP de LightGBM."""
    params = parameters.copy()
    params['model_name'] = 'LightGBM'
    return shap_analysis_nodes.calculate_shap_values_classification(model_result, train_data, test_data, params)


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
            # Análisis SHAP para interpretabilidad avanzada
            node(
                func=_calculate_rf_shap,
                inputs=["rf_clf_result", "train_data", "test_data", "params:classification"],
                outputs="rf_clf_shap",
                name="calculate_rf_shap_node",
                tags=["classification", "shap", "interpretability"],
            ),
            node(
                func=_calculate_xgb_shap,
                inputs=["xgb_clf_result", "train_data", "test_data", "params:classification"],
                outputs="xgb_clf_shap",
                name="calculate_xgb_shap_node",
                tags=["classification", "shap", "interpretability"],
            ),
            node(
                func=_calculate_lgbm_shap,
                inputs=["lgbm_clf_result", "train_data", "test_data", "params:classification"],
                outputs="lgbm_clf_shap",
                name="calculate_lgbm_shap_node",
                tags=["classification", "shap", "interpretability"],
            ),
            node(
                func=shap_analysis_nodes.create_shap_summary_classification,
                inputs=["rf_clf_shap", "xgb_clf_shap", "lgbm_clf_shap"],
                outputs="classification_shap_summary",
                name="create_classification_shap_summary_node",
                tags=["classification", "shap", "summary"],
            ),
        ]
    )
