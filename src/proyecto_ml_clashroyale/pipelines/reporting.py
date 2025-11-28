"""Pipeline para generar reportes consolidados."""

import json
from pathlib import Path
from kedro.pipeline import Pipeline, node
from .nodes.reporting_nodes import generate_models_report


def _load_optional_from_catalog_or_file(catalog, dataset_name: str, default_path: Path = None):
    """Cargar dataset del catálogo si existe, sino intentar desde archivo.
    
    Args:
        catalog: Catálogo de Kedro
        dataset_name: Nombre del dataset en el catálogo
        default_path: Ruta alternativa al archivo si no está en catálogo
        
    Returns:
        Datos del dataset o None
    """
    # Intentar desde catálogo
    try:
        if hasattr(catalog, 'load'):
            return catalog.load(dataset_name)
    except Exception:
        pass
    
    # Intentar desde archivo si se proporciona ruta
    if default_path and default_path.exists():
        try:
            with open(default_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    return None


def generate_models_report_from_catalog(
    classification_comparison=None,
    regression_comparison=None,
    clustering_comparison=None,
    pca_metrics=None,
    umap_metrics=None,
    anomaly_detection_comparison=None,
    association_rules_comparison=None,
    classification_shap_summary=None,
    regression_shap_summary=None,
) -> dict:
    """Generar informe consolidado desde inputs del catálogo (pueden ser None).
    
    Args:
        Todos los inputs son opcionales y pueden ser None
        
    Returns:
        Diccionario con informe consolidado
    """
    return generate_models_report(
        classification_comparison=classification_comparison,
        regression_comparison=regression_comparison,
        clustering_comparison=clustering_comparison,
        pca_metrics=pca_metrics,
        umap_metrics=umap_metrics,
        anomaly_detection_comparison=anomaly_detection_comparison,
        association_rules_comparison=association_rules_comparison,
        classification_shap_summary=classification_shap_summary,
        regression_shap_summary=regression_shap_summary,
    )


def create_reporting_pipeline() -> Pipeline:
    """Crear pipeline para generar reportes consolidados.
    
    Nota: Este pipeline intenta cargar todos los datasets disponibles.
    Si algún dataset no está disponible, se genera el informe con los disponibles.
    Alternativamente, se puede usar el script independiente:
    python scripts/generate_models_report.py
    
    Returns:
        Pipeline de reporting
    """
    return Pipeline(
        [
            node(
                func=generate_models_report_from_catalog,
                inputs={
                    "classification_comparison": "classification_comparison",
                    "regression_comparison": "regression_comparison",
                    "clustering_comparison": "clustering_comparison",
                    "pca_metrics": "pca_metrics",
                    "umap_metrics": "umap_metrics",
                    "anomaly_detection_comparison": "anomaly_detection_comparison",
                    "association_rules_comparison": "association_rules_comparison",
                    "classification_shap_summary": "classification_shap_summary",
                    "regression_shap_summary": "regression_shap_summary",
                },
                outputs="models_report",
                name="generate_models_report_node",
                tags=["reporting", "models_report"],
            ),
        ]
    )

