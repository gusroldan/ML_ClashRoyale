"""Nodos para integrar clustering como feature engineering en modelos supervisados."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
import pickle

logger = logging.getLogger(__name__)


def add_cluster_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    clustering_result: Dict[str, Any],
    cluster_model_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Agregar features de clustering a los datos de entrenamiento y test.
    
    Args:
        train_data: Datos de entrenamiento
        test_data: Datos de test
        clustering_result: Resultado del modelo de clustering
        cluster_model_name: Nombre del modelo de clustering
        
    Returns:
        Tupla con (train_data_con_clusters, test_data_con_clusters)
    """
    logger.info(f"Agregando features de clustering ({cluster_model_name}) a datos de entrenamiento y test...")
    
    # Obtener modelo y scaler del resultado
    model = clustering_result['model']
    scaler = clustering_result.get('scaler', None)
    
    # Extraer features (sin labels)
    feature_cols = [col for col in train_data.columns 
                   if col not in ['label', 'target_regression', 'battle_id']]
    X_train = train_data[feature_cols].copy()
    X_test = test_data[feature_cols].copy()
    
    # Normalizar si hay scaler
    if scaler is not None:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
    
    # Predecir clusters
    train_clusters = model.predict(X_train_scaled) if hasattr(model, 'predict') else model.fit_predict(X_train_scaled)
    test_clusters = model.predict(X_test_scaled) if hasattr(model, 'predict') else model.fit_predict(X_test_scaled)
    
    # Agregar features de cluster
    train_data_with_clusters = train_data.copy()
    test_data_with_clusters = test_data.copy()
    
    # Feature principal: cluster asignado
    train_data_with_clusters[f'cluster_{cluster_model_name}'] = train_clusters
    test_data_with_clusters[f'cluster_{cluster_model_name}'] = test_clusters
    
    # Features adicionales: one-hot encoding de clusters (solo si hay pocos clusters)
    n_clusters = len(np.unique(train_clusters))
    if n_clusters <= 20:  # Solo hacer one-hot si hay menos de 20 clusters
        for cluster_id in range(n_clusters):
            train_data_with_clusters[f'cluster_{cluster_model_name}_{cluster_id}'] = (train_clusters == cluster_id).astype(int)
            test_data_with_clusters[f'cluster_{cluster_model_name}_{cluster_id}'] = (test_clusters == cluster_id).astype(int)
    
    logger.info(f"Features de clustering agregadas: {n_clusters} clusters identificados")
    logger.info(f"Train: {train_data_with_clusters.shape}, Test: {test_data_with_clusters.shape}")
    
    return train_data_with_clusters, test_data_with_clusters


def evaluate_cluster_features_improvement(
    baseline_metrics: Dict[str, Any],
    enhanced_metrics: Dict[str, Any],
    model_name: str
) -> Dict[str, Any]:
    """Evaluar la mejora obtenida al agregar features de clustering.
    
    Args:
        baseline_metrics: Métricas del modelo sin features de clustering
        enhanced_metrics: Métricas del modelo con features de clustering
        model_name: Nombre del modelo
        
    Returns:
        Diccionario con análisis de mejora
    """
    logger.info(f"Evaluando mejora de features de clustering para {model_name}...")
    
    improvement = {}
    
    # Comparar métricas de clasificación
    if 'accuracy' in baseline_metrics and 'accuracy' in enhanced_metrics:
        baseline_acc = baseline_metrics['accuracy']
        enhanced_acc = enhanced_metrics['accuracy']
        improvement['accuracy'] = {
            'baseline': float(baseline_acc),
            'enhanced': float(enhanced_acc),
            'improvement': float(enhanced_acc - baseline_acc),
            'improvement_percentage': float((enhanced_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0.0
        }
    
    if 'roc_auc' in baseline_metrics and 'roc_auc' in enhanced_metrics:
        baseline_auc = baseline_metrics['roc_auc']
        enhanced_auc = enhanced_metrics['roc_auc']
        improvement['roc_auc'] = {
            'baseline': float(baseline_auc),
            'enhanced': float(enhanced_auc),
            'improvement': float(enhanced_auc - baseline_auc),
            'improvement_percentage': float((enhanced_auc - baseline_auc) / baseline_auc * 100) if baseline_auc > 0 else 0.0
        }
    
    # Comparar métricas de regresión
    if 'r2_score' in baseline_metrics and 'r2_score' in enhanced_metrics:
        baseline_r2 = baseline_metrics['r2_score']
        enhanced_r2 = enhanced_metrics['r2_score']
        improvement['r2_score'] = {
            'baseline': float(baseline_r2),
            'enhanced': float(enhanced_r2),
            'improvement': float(enhanced_r2 - baseline_r2),
            'improvement_percentage': float((enhanced_r2 - baseline_r2) / abs(baseline_r2) * 100) if baseline_r2 != 0 else 0.0
        }
    
    if 'rmse' in baseline_metrics and 'rmse' in enhanced_metrics:
        baseline_rmse = baseline_metrics['rmse']
        enhanced_rmse = enhanced_metrics['rmse']
        improvement['rmse'] = {
            'baseline': float(baseline_rmse),
            'enhanced': float(enhanced_rmse),
            'improvement': float(baseline_rmse - enhanced_rmse),  # RMSE menor es mejor
            'improvement_percentage': float((baseline_rmse - enhanced_rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0.0
        }
    
    # Resumen
    improvement['summary'] = {
        'model_name': model_name,
        'overall_improvement': 'positive' if any(
            v.get('improvement', 0) > 0 if 'rmse' not in k else v.get('improvement', 0) > 0
            for k, v in improvement.items() if isinstance(v, dict) and 'improvement' in v
        ) else 'neutral',
        'recommendation': 'Usar features de clustering' if improvement.get('summary', {}).get('overall_improvement') == 'positive' else 'Evaluar caso por caso'
    }
    
    logger.info(f"Análisis de mejora completado para {model_name}")
    
    return improvement

