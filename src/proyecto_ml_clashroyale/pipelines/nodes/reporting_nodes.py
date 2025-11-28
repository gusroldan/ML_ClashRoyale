"""Nodos para generar reportes consolidados del proyecto."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def generate_models_report(
    classification_comparison: Dict[str, Any] = None,
    regression_comparison: Dict[str, Any] = None,
    clustering_comparison: Dict[str, Any] = None,
    pca_metrics: Dict[str, Any] = None,
    umap_metrics: Dict[str, Any] = None,
    anomaly_detection_comparison: Dict[str, Any] = None,
    association_rules_comparison: Dict[str, Any] = None,
    classification_shap_summary: Dict[str, Any] = None,
    regression_shap_summary: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Generar informe consolidado de todos los modelos entrenados.
    
    Args:
        classification_comparison: Comparación de modelos de clasificación
        regression_comparison: Comparación de modelos de regresión
        clustering_comparison: Comparación de algoritmos de clustering
        pca_metrics: Métricas de PCA
        umap_metrics: Métricas de UMAP (opcional)
        anomaly_detection_comparison: Comparación de detección de anomalías (opcional)
        association_rules_comparison: Comparación de reglas de asociación (opcional)
        classification_shap_summary: Resumen SHAP de clasificación (opcional)
        regression_shap_summary: Resumen SHAP de regresión (opcional)
        
    Returns:
        Diccionario con informe consolidado en formato JSON
    """
    logger.info("Generando informe consolidado de modelos entrenados...")
    
    report = {
        "metadata": {
            "project_name": "ML ClashRoyale",
            "report_type": "Modelos Entrenados - Informe Consolidado",
            "generation_date": datetime.now().isoformat(),
            "version": "1.0"
        },
        "summary": {
            "total_models": 0,
            "supervised_models": 0,
            "unsupervised_models": 0,
            "best_models": {}
        },
        "supervised_learning": {
            "classification": {},
            "regression": {}
        },
        "unsupervised_learning": {
            "clustering": {},
            "dimensionality_reduction": {},
            "anomaly_detection": {},
            "association_rules": {}
        },
        "interpretability": {
            "shap_analysis": {}
        }
    }
    
    # ========== CLASIFICACIÓN ==========
    if classification_comparison:
        logger.info("Procesando modelos de clasificación...")
        classification_models = {}
        
        if 'models' in classification_comparison:
            models_dict = classification_comparison['models']
            for model_name, model_data in models_dict.items():
                if isinstance(model_data, dict) and 'metrics' in model_data:
                    metrics = model_data['metrics']
                    classification_models[model_name] = {
                        "accuracy": float(metrics.get('accuracy', 0)),
                        "precision": float(metrics.get('precision', 0)),
                        "recall": float(metrics.get('recall', 0)),
                        "f1_score": float(metrics.get('f1', 0)),
                        "roc_auc": float(metrics.get('roc_auc', 0)),
                        "best_params": model_data.get('best_params', {})
                    }
        
        # Identificar mejor modelo de clasificación
        if classification_models:
            best_clf = max(classification_models.items(), key=lambda x: x[1]['roc_auc'])
            report["summary"]["best_models"]["classification"] = {
                "model": best_clf[0],
                "roc_auc": best_clf[1]['roc_auc'],
                "accuracy": best_clf[1]['accuracy']
            }
        
        report["supervised_learning"]["classification"] = {
            "models": classification_models,
            "total_models": len(classification_models),
            "task": "Predecir si el jugador A gana o pierde la batalla (binario 1/0)"
        }
        report["summary"]["supervised_models"] += len(classification_models)
    
    # ========== REGRESIÓN ==========
    if regression_comparison:
        logger.info("Procesando modelos de regresión...")
        regression_models = {}
        
        if 'models' in regression_comparison:
            models_dict = regression_comparison['models']
            for model_name, model_data in models_dict.items():
                if isinstance(model_data, dict) and 'metrics' in model_data:
                    metrics = model_data['metrics']
                    regression_models[model_name] = {
                        "mae": float(metrics.get('mae', 0)),
                        "mse": float(metrics.get('mse', 0)),
                        "rmse": float(metrics.get('rmse', 0)),
                        "r2": float(metrics.get('r2', 0)),
                        "best_params": model_data.get('best_params', {})
                    }
        
        # Identificar mejor modelo de regresión
        if regression_models:
            best_reg = min(regression_models.items(), key=lambda x: x[1]['mae'])
            report["summary"]["best_models"]["regression"] = {
                "model": best_reg[0],
                "mae": best_reg[1]['mae'],
                "r2": best_reg[1]['r2']
            }
        
        report["supervised_learning"]["regression"] = {
            "models": regression_models,
            "total_models": len(regression_models),
            "task": "Predecir el cambio de trofeos del jugador A (A.trophyChange, con signo)"
        }
        report["summary"]["supervised_models"] += len(regression_models)
    
    # ========== CLUSTERING ==========
    if clustering_comparison:
        logger.info("Procesando algoritmos de clustering...")
        clustering_algorithms = {}
        
        for algo_name in ['kmeans', 'optics', 'hierarchical']:
            if algo_name in clustering_comparison:
                algo_data = clustering_comparison[algo_name]
                clustering_algorithms[algo_name] = {
                    "silhouette_score": float(algo_data.get('silhouette_score', 0)),
                    "davies_bouldin_index": float(algo_data.get('davies_bouldin_index', 0)),
                    "calinski_harabasz_index": float(algo_data.get('calinski_harabasz_index', 0)),
                    "n_clusters": int(algo_data.get('n_clusters', 0)),
                    "n_noise": int(algo_data.get('n_noise', 0)) if algo_name == 'optics' else None
                }
        
        # Identificar mejor algoritmo de clustering
        if clustering_algorithms:
            best_clustering = max(
                clustering_algorithms.items(),
                key=lambda x: x[1]['silhouette_score']
            )
            report["summary"]["best_models"]["clustering"] = {
                "algorithm": best_clustering[0],
                "silhouette_score": best_clustering[1]['silhouette_score'],
                "n_clusters": best_clustering[1]['n_clusters']
            }
        
        report["unsupervised_learning"]["clustering"] = {
            "algorithms": clustering_algorithms,
            "total_algorithms": len(clustering_algorithms),
            "task": "Identificar grupos naturales de batallas similares"
        }
        report["summary"]["unsupervised_models"] += len(clustering_algorithms)
    
    # ========== REDUCCIÓN DE DIMENSIONALIDAD ==========
    if pca_metrics:
        logger.info("Procesando métricas de PCA...")
        pca_info = {
            "n_components": int(pca_metrics.get('n_components', 0)),
            "explained_variance": {
                "total": float(pca_metrics.get('explained_variance', {}).get('total_explained_variance', 0)),
                "per_component": [float(v) for v in pca_metrics.get('explained_variance', {}).get('per_component', [])]
            }
        }
        report["unsupervised_learning"]["dimensionality_reduction"]["pca"] = pca_info
        report["summary"]["unsupervised_models"] += 1
    
    if umap_metrics:
        logger.info("Procesando métricas de UMAP...")
        umap_info = {
            "n_components": int(umap_metrics.get('n_components', 2)),
            "n_neighbors": int(umap_metrics.get('n_neighbors', 15)),
            "min_dist": float(umap_metrics.get('min_dist', 0.1)),
            "metric": umap_metrics.get('metric', 'euclidean')
        }
        report["unsupervised_learning"]["dimensionality_reduction"]["umap"] = umap_info
        report["summary"]["unsupervised_models"] += 1
    
    # ========== DETECCIÓN DE ANOMALÍAS ==========
    if anomaly_detection_comparison:
        logger.info("Procesando algoritmos de detección de anomalías...")
        anomaly_algorithms = {}
        
        for algo_name, algo_data in anomaly_detection_comparison.items():
            if algo_name not in ['summary', 'best_model'] and isinstance(algo_data, dict):
                if algo_data.get('available', True):
                    anomaly_algorithms[algo_name] = {
                        "n_anomalies": int(algo_data.get('n_anomalies', 0)),
                        "anomaly_percentage": float(algo_data.get('anomaly_percentage', 0)),
                        "mean_score": float(algo_data.get('mean_score', 0)) if algo_data.get('mean_score') else None,
                        "threshold": float(algo_data.get('threshold', 0)) if algo_data.get('threshold') else None
                    }
        
        # Identificar mejor algoritmo (más balanceado)
        if anomaly_algorithms:
            best_anomaly = min(
                anomaly_algorithms.items(),
                key=lambda x: abs(x[1]['anomaly_percentage'] - 10)
            )
            report["summary"]["best_models"]["anomaly_detection"] = {
                "algorithm": best_anomaly[0],
                "n_anomalies": best_anomaly[1]['n_anomalies'],
                "anomaly_percentage": best_anomaly[1]['anomaly_percentage']
            }
        
        report["unsupervised_learning"]["anomaly_detection"] = {
            "algorithms": anomaly_algorithms,
            "total_algorithms": len(anomaly_algorithms),
            "task": "Identificar batallas o patrones inusuales"
        }
        report["summary"]["unsupervised_models"] += len(anomaly_algorithms)
    
    # ========== REGLAS DE ASOCIACIÓN ==========
    if association_rules_comparison:
        logger.info("Procesando algoritmos de reglas de asociación...")
        association_algorithms = {}
        
        apriori_data = association_rules_comparison.get('apriori', {})
        fpgrowth_data = association_rules_comparison.get('fpgrowth', {})
        
        if apriori_data:
            association_algorithms['apriori'] = {
                "n_itemsets": int(apriori_data.get('n_itemsets', 0)),
                "n_rules": int(apriori_data.get('n_rules', 0)),
                "avg_lift": float(apriori_data.get('avg_lift', 0)) if apriori_data.get('avg_lift') is not None else None,
                "avg_confidence": float(apriori_data.get('avg_confidence', 0)) if apriori_data.get('avg_confidence') is not None else None
            }
        
        if fpgrowth_data:
            association_algorithms['fpgrowth'] = {
                "n_itemsets": int(fpgrowth_data.get('n_itemsets', 0)),
                "n_rules": int(fpgrowth_data.get('n_rules', 0)),
                "avg_lift": float(fpgrowth_data.get('avg_lift', 0)) if fpgrowth_data.get('avg_lift') is not None else None,
                "avg_confidence": float(fpgrowth_data.get('avg_confidence', 0)) if fpgrowth_data.get('avg_confidence') is not None else None
            }
        
        # Identificar mejor algoritmo
        if association_algorithms:
            best_association = max(
                association_algorithms.items(),
                key=lambda x: x[1]['n_rules'] if x[1]['n_rules'] else 0
            )
            report["summary"]["best_models"]["association_rules"] = {
                "algorithm": best_association[0],
                "n_rules": best_association[1]['n_rules'],
                "n_itemsets": best_association[1]['n_itemsets']
            }
        
        report["unsupervised_learning"]["association_rules"] = {
            "algorithms": association_algorithms,
            "total_algorithms": len(association_algorithms),
            "task": "Descubrir relaciones frecuentes entre cartas"
        }
        report["summary"]["unsupervised_models"] += len(association_algorithms)
    
    # ========== SHAP ANALYSIS ==========
    if classification_shap_summary:
        logger.info("Procesando resumen SHAP de clasificación...")
        report["interpretability"]["shap_analysis"]["classification"] = {
            "models_analyzed": classification_shap_summary.get('models_analyzed', []),
            "top_features_across_models": classification_shap_summary.get('top_features_across_models', {}),
            "model_comparison": classification_shap_summary.get('model_comparison', [])
        }
    
    if regression_shap_summary:
        logger.info("Procesando resumen SHAP de regresión...")
        report["interpretability"]["shap_analysis"]["regression"] = {
            "models_analyzed": regression_shap_summary.get('models_analyzed', []),
            "top_features_across_models": regression_shap_summary.get('top_features_across_models', {}),
            "model_comparison": regression_shap_summary.get('model_comparison', [])
        }
    
    # Calcular total de modelos
    report["summary"]["total_models"] = (
        report["summary"]["supervised_models"] + 
        report["summary"]["unsupervised_models"]
    )
    
    logger.info(f"Informe consolidado generado: {report['summary']['total_models']} modelos totales")
    
    return report

