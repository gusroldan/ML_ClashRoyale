"""Nodos para el pipeline de clustering (aprendizaje no supervisado)."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import logging
import json

logger = logging.getLogger(__name__)


def _extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """Extraer solo las features del dataset (sin labels).
    
    Args:
        data: DataFrame con features y labels
        
    Returns:
        DataFrame solo con features
    """
    # Eliminar columnas de labels si existen
    columns_to_drop = ['label', 'target_regression']
    X = data.drop(columns=[col for col in columns_to_drop if col in data.columns]).copy()
    return X


def _calculate_elbow_method(X: pd.DataFrame, k_range: List[int], random_state: int) -> Dict[str, Any]:
    """Calcular el método del codo para K-Means.
    
    Args:
        X: Datos normalizados
        k_range: Rango de valores de k a probar
        random_state: Semilla aleatoria
        
    Returns:
        Diccionario con k_values e inercias
    """
    inercias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inercias.append(float(kmeans.inertia_))
    
    return {
        'k_values': [int(k) for k in k_range],
        'inertias': inercias
    }


def _calculate_dendrogram_data(X: pd.DataFrame, method: str = 'ward', metric: str = 'euclidean') -> Dict[str, Any]:
    """Calcular datos para generar dendrograma (solo para Hierarchical Clustering).
    
    Args:
        X: Datos normalizados
        method: Método de linkage ('ward', 'complete', 'average', 'single')
        metric: Métrica de distancia ('euclidean', 'manhattan', etc.)
        
    Returns:
        Diccionario con datos del dendrograma
    """
    # Calcular linkage matrix
    linkage_matrix = linkage(X, method=method, metric=metric)
    
    # Extraer información relevante del dendrograma
    # Nota: No podemos guardar el objeto completo del dendrograma, pero podemos guardar
    # la información necesaria para recrearlo
    return {
        'linkage_matrix': linkage_matrix.tolist(),  # Convertir a lista para JSON
        'method': method,
        'metric': metric,
        'n_samples': int(X.shape[0])
    }


def save_clustering_metrics(metrics: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Guardar métricas del clustering en formato serializable.
    
    Args:
        metrics: Diccionario con métricas
        model_name: Nombre del modelo
        
    Returns:
        Diccionario con métricas convertidas a tipos básicos
    """
    result = {
        'model_name': model_name,
        'n_clusters': int(metrics.get('n_clusters', 0)),
        'silhouette_score': float(metrics.get('silhouette_score', 0.0)),
        'davies_bouldin_index': float(metrics.get('davies_bouldin_index', 0.0)),
        'calinski_harabasz_index': float(metrics.get('calinski_harabasz_index', 0.0)),
        'parameters': metrics.get('parameters', {})
    }
    
    # Agregar métricas específicas si existen
    if 'elbow_method' in metrics:
        result['elbow_method'] = metrics['elbow_method']
    
    if 'dendrogram_data' in metrics:
        result['dendrogram_data'] = metrics['dendrogram_data']
    
    return result


def train_kmeans(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de K-Means clustering.
    
    Args:
        train_data: Data de entrenamiento
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (resultado del modelo, métricas en formato JSON)
    """
    logger.info("Entrenando K-Means clustering...")
    
    X = _extract_features(train_data)
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Parámetros
    n_clusters = parameters['models']['kmeans']['n_clusters']
    random_state = parameters['random_state']
    max_iter = parameters['models']['kmeans'].get('max_iter', 300)
    n_init = parameters['models']['kmeans'].get('n_init', 10)
    
    # Entrenar K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init
    )
    kmeans.fit(X_scaled_df)
    
    # Predecir clusters
    labels = kmeans.predict(X_scaled_df)
    
    # Calcular métricas
    silhouette = silhouette_score(X_scaled_df, labels)
    davies_bouldin = davies_bouldin_score(X_scaled_df, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled_df, labels)
    
    # Calcular método del codo si está configurado
    elbow_method = None
    if parameters['models']['kmeans'].get('calculate_elbow', False):
        k_range = parameters['models']['kmeans'].get('elbow_k_range', list(range(2, 11)))
        elbow_method = _calculate_elbow_method(X_scaled_df, k_range, random_state)
    
    metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_index': calinski_harabasz,
        'parameters': {
            'n_clusters': n_clusters,
            'max_iter': max_iter,
            'n_init': n_init,
            'random_state': random_state
        }
    }
    
    if elbow_method:
        metrics['elbow_method'] = elbow_method
    
    logger.info(f"K-Means - Clusters: {n_clusters}, Silhouette: {silhouette:.4f}, "
                f"Davies-Bouldin: {davies_bouldin:.4f}, Calinski-Harabasz: {calinski_harabasz:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_clustering_metrics(metrics, 'K-Means')
    
    return {
        'model': kmeans,
        'scaler': scaler,
        'labels': labels,
        'metrics': metrics
    }, metrics_json


def train_optics(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de OPTICS clustering.
    
    Args:
        train_data: Data de entrenamiento
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (resultado del modelo, métricas en formato JSON)
    """
    logger.info("Entrenando OPTICS clustering...")
    
    X = _extract_features(train_data)
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Parámetros
    min_samples = parameters['models']['optics'].get('min_samples', 5)
    max_eps = parameters['models']['optics'].get('max_eps', None)
    if max_eps is None:
        max_eps = np.inf
    metric = parameters['models']['optics'].get('metric', 'euclidean')
    n_jobs = parameters['models']['optics'].get('n_jobs', -1)
    
    # Entrenar OPTICS
    optics = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        metric=metric,
        n_jobs=n_jobs
    )
    optics.fit(X_scaled_df)
    
    # Predecir clusters
    labels = optics.labels_
    
    # Filtrar puntos de ruido (label = -1) para métricas
    # Si hay demasiados puntos de ruido, las métricas pueden no ser confiables
    non_noise_mask = labels != -1
    n_noise = np.sum(~non_noise_mask)
    n_clusters = len(set(labels[non_noise_mask])) if np.any(non_noise_mask) else 0
    
    if n_clusters < 2:
        logger.warning(f"OPTICS encontró {n_clusters} clusters válidos. Usando todos los puntos para métricas.")
        non_noise_mask = np.ones(len(labels), dtype=bool)
        n_clusters = len(set(labels)) if len(set(labels)) > 1 else 1
    
    # Calcular métricas solo con puntos no-ruido si hay suficientes
    if np.sum(non_noise_mask) > 10:
        X_clean = X_scaled_df[non_noise_mask]
        labels_clean = labels[non_noise_mask]
        
        silhouette = silhouette_score(X_clean, labels_clean)
        davies_bouldin = davies_bouldin_score(X_clean, labels_clean)
        calinski_harabasz = calinski_harabasz_score(X_clean, labels_clean)
    else:
        # Si hay muy pocos puntos no-ruido, usar todos
        silhouette = silhouette_score(X_scaled_df, labels)
        davies_bouldin = davies_bouldin_score(X_scaled_df, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled_df, labels)
    
    metrics = {
        'n_clusters': int(n_clusters),
        'n_noise_points': int(n_noise),
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin),
        'calinski_harabasz_index': float(calinski_harabasz),
        'parameters': {
            'min_samples': min_samples,
            'max_eps': float(max_eps) if max_eps != np.inf else 'inf',
            'metric': metric
        }
    }
    
    logger.info(f"OPTICS - Clusters: {n_clusters}, Noise points: {n_noise}, "
                f"Silhouette: {silhouette:.4f}, Davies-Bouldin: {davies_bouldin:.4f}, "
                f"Calinski-Harabasz: {calinski_harabasz:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_clustering_metrics(metrics, 'OPTICS')
    
    return {
        'model': optics,
        'scaler': scaler,
        'labels': labels,
        'metrics': metrics
    }, metrics_json


def train_hierarchical_clustering(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de Hierarchical Clustering.
    
    Args:
        train_data: Data de entrenamiento
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (resultado del modelo, métricas en formato JSON)
    """
    logger.info("Entrenando Hierarchical Clustering...")
    
    X = _extract_features(train_data)
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Parámetros
    n_clusters = parameters['models']['hierarchical']['n_clusters']
    linkage_method = parameters['models']['hierarchical'].get('linkage', 'ward')
    metric = parameters['models']['hierarchical'].get('metric', 'euclidean')
    
    # Entrenar Hierarchical Clustering
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        metric=metric if linkage_method != 'ward' else 'euclidean'
    )
    labels = hierarchical.fit_predict(X_scaled_df)
    
    # Calcular métricas
    silhouette = silhouette_score(X_scaled_df, labels)
    davies_bouldin = davies_bouldin_score(X_scaled_df, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled_df, labels)
    
    # Calcular datos del dendrograma si está configurado
    dendrogram_data = None
    if parameters['models']['hierarchical'].get('calculate_dendrogram', False):
        # Para muestras grandes, puede ser lento, así que podemos submuestrear
        max_samples_dendrogram = parameters['models']['hierarchical'].get('max_samples_dendrogram', 5000)
        if len(X_scaled_df) > max_samples_dendrogram:
            logger.info(f"Submuestreando a {max_samples_dendrogram} muestras para dendrograma...")
            indices = np.random.choice(len(X_scaled_df), max_samples_dendrogram, replace=False)
            X_dendro = X_scaled_df.iloc[indices]
        else:
            X_dendro = X_scaled_df
        
        dendrogram_data = _calculate_dendrogram_data(X_dendro, method=linkage_method, metric=metric)
    
    metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_index': calinski_harabasz,
        'parameters': {
            'n_clusters': n_clusters,
            'linkage': linkage_method,
            'metric': metric
        }
    }
    
    if dendrogram_data:
        metrics['dendrogram_data'] = dendrogram_data
    
    logger.info(f"Hierarchical Clustering - Clusters: {n_clusters}, Silhouette: {silhouette:.4f}, "
                f"Davies-Bouldin: {davies_bouldin:.4f}, Calinski-Harabasz: {calinski_harabasz:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_clustering_metrics(metrics, 'Hierarchical Clustering')
    
    return {
        'model': hierarchical,
        'scaler': scaler,
        'labels': labels,
        'metrics': metrics
    }, metrics_json


def create_clustering_comparison(
    kmeans_result: Dict[str, Any],
    optics_result: Dict[str, Any],
    hierarchical_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Crear comparación entre los diferentes métodos de clustering.
    
    Args:
        kmeans_result: Resultado de K-Means
        optics_result: Resultado de OPTICS
        hierarchical_result: Resultado de Hierarchical Clustering
        
    Returns:
        Diccionario con comparación de modelos
    """
    logger.info("Creando comparación de modelos de clustering...")
    
    comparison = {
        'kmeans': {
            'silhouette_score': float(kmeans_result['metrics']['silhouette_score']),
            'davies_bouldin_index': float(kmeans_result['metrics']['davies_bouldin_index']),
            'calinski_harabasz_index': float(kmeans_result['metrics']['calinski_harabasz_index']),
            'n_clusters': int(kmeans_result['metrics']['n_clusters'])
        },
        'optics': {
            'silhouette_score': float(optics_result['metrics']['silhouette_score']),
            'davies_bouldin_index': float(optics_result['metrics']['davies_bouldin_index']),
            'calinski_harabasz_index': float(optics_result['metrics']['calinski_harabasz_index']),
            'n_clusters': int(optics_result['metrics']['n_clusters']),
            'n_noise_points': int(optics_result['metrics'].get('n_noise_points', 0))
        },
        'hierarchical': {
            'silhouette_score': float(hierarchical_result['metrics']['silhouette_score']),
            'davies_bouldin_index': float(hierarchical_result['metrics']['davies_bouldin_index']),
            'calinski_harabasz_index': float(hierarchical_result['metrics']['calinski_harabasz_index']),
            'n_clusters': int(hierarchical_result['metrics']['n_clusters'])
        }
    }
    
    # Determinar mejor modelo por cada métrica
    # Silhouette: mayor es mejor
    best_silhouette = max(
        comparison['kmeans']['silhouette_score'],
        comparison['optics']['silhouette_score'],
        comparison['hierarchical']['silhouette_score']
    )
    
    # Davies-Bouldin: menor es mejor
    best_davies_bouldin = min(
        comparison['kmeans']['davies_bouldin_index'],
        comparison['optics']['davies_bouldin_index'],
        comparison['hierarchical']['davies_bouldin_index']
    )
    
    # Calinski-Harabasz: mayor es mejor
    best_calinski_harabasz = max(
        comparison['kmeans']['calinski_harabasz_index'],
        comparison['optics']['calinski_harabasz_index'],
        comparison['hierarchical']['calinski_harabasz_index']
    )
    
    comparison['best_models'] = {
        'silhouette_score': {
            'model': 'kmeans' if comparison['kmeans']['silhouette_score'] == best_silhouette else
                     ('optics' if comparison['optics']['silhouette_score'] == best_silhouette else 'hierarchical'),
            'score': float(best_silhouette)
        },
        'davies_bouldin_index': {
            'model': 'kmeans' if comparison['kmeans']['davies_bouldin_index'] == best_davies_bouldin else
                     ('optics' if comparison['optics']['davies_bouldin_index'] == best_davies_bouldin else 'hierarchical'),
            'score': float(best_davies_bouldin)
        },
        'calinski_harabasz_index': {
            'model': 'kmeans' if comparison['kmeans']['calinski_harabasz_index'] == best_calinski_harabasz else
                     ('optics' if comparison['optics']['calinski_harabasz_index'] == best_calinski_harabasz else 'hierarchical'),
            'score': float(best_calinski_harabasz)
        }
    }
    
    logger.info("Comparación de clustering completada")
    
    return comparison

