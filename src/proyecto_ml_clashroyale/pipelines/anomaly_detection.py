"""Pipeline para detección de anomalías."""
## Objetivo: identificar puntos anómalos o outliers en los datos.
from kedro.pipeline import Pipeline, node
from typing import Dict
from proyecto_ml_clashroyale.pipelines.nodes import anomaly_detection_nodes


def create_anomaly_detection_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de detección de anomalías.
    
    Returns:
        Pipeline de detección de anomalías
    """
    
    nodes_list = [
        node(
            func=anomaly_detection_nodes.detect_anomalies_isolation_forest,
            inputs=["train_data", "params:anomaly_detection"],
            outputs=["isolation_forest_result", "isolation_forest_metrics"],
            name="detect_anomalies_isolation_forest_node",
            tags=["anomaly_detection", "isolation_forest"],
        ),
        node(
            func=anomaly_detection_nodes.detect_anomalies_lof,
            inputs=["train_data", "params:anomaly_detection"],
            outputs=["lof_result", "lof_metrics"],
            name="detect_anomalies_lof_node",
            tags=["anomaly_detection", "lof"],
        ),
        node(
            func=anomaly_detection_nodes.detect_anomalies_oneclass_svm,
            inputs=["train_data", "params:anomaly_detection"],
            outputs=["oneclass_svm_result", "oneclass_svm_metrics"],
            name="detect_anomalies_oneclass_svm_node",
            tags=["anomaly_detection", "oneclass_svm"],
        ),
    ]
    
    # Agregar nodo de autoencoder solo si TensorFlow está disponible
    # Si no está disponible, crear un nodo que retorne None
    try:
        import tensorflow as tf
        nodes_list.append(
            node(
                func=anomaly_detection_nodes.detect_anomalies_autoencoder,
                inputs=["train_data", "params:anomaly_detection"],
                outputs=["autoencoder_result", "autoencoder_metrics"],
                name="detect_anomalies_autoencoder_node",
                tags=["anomaly_detection", "autoencoder"],
            )
        )
    except ImportError:
        # Crear nodo dummy que retorne un diccionario válido si TensorFlow no está disponible
        def dummy_autoencoder(*args, **kwargs):
            # Retornar estructura válida en lugar de None
            dummy_result = {
                'model': None,
                'scaler': None,
                'predictions': None,
                'anomaly_labels': None,
                'scores': None,
                'metrics': {
                    'n_anomalies': 0,
                    'n_normal': 0,
                    'anomaly_percentage': 0.0,
                    'scores': [],
                    'parameters': {},
                    'available': False,
                    'message': 'TensorFlow no disponible'
                }
            }
            dummy_metrics = {
                'model_name': 'Autoencoder',
                'n_anomalies': 0,
                'n_normal': 0,
                'anomaly_percentage': 0.0,
                'message': 'TensorFlow no disponible',
                'available': False
            }
            return dummy_result, dummy_metrics
        
        nodes_list.append(
            node(
                func=dummy_autoencoder,
                inputs=["train_data", "params:anomaly_detection"],
                outputs=["autoencoder_result", "autoencoder_metrics"],
                name="detect_anomalies_autoencoder_node",
                tags=["anomaly_detection", "autoencoder"],
            )
        )
    
    # Nodo de comparación (siempre incluye autoencoder_result, puede ser None)
    nodes_list.append(
        node(
            func=anomaly_detection_nodes.create_anomaly_detection_comparison,
            inputs=["isolation_forest_result", "lof_result", "oneclass_svm_result", "autoencoder_result"],
            outputs="anomaly_detection_comparison",
            name="create_anomaly_detection_comparison_node",
            tags=["anomaly_detection", "evaluation"],
        )
    )
    
    return Pipeline(nodes_list)

