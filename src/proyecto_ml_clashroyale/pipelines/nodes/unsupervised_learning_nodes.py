"""Nodos para el pipeline de clustering (aprendizaje no supervisado)."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os

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


def generate_clustering_visualizations(
    train_data: pd.DataFrame,
    kmeans_result: Dict[str, Any],
    optics_result: Dict[str, Any],
    hierarchical_result: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, str]:
    """Generar y guardar visualizaciones de clustering en formato PNG.
    
    Args:
        train_data: Datos de entrenamiento
        kmeans_result: Resultado de K-Means
        optics_result: Resultado de OPTICS
        hierarchical_result: Resultado de Hierarchical Clustering
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con las rutas de los archivos PNG generados
    """
    logger.info("Generando visualizaciones de clustering...")
    
    # Configurar estilo
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    sns.set_palette("husl")
    
    output_dir = parameters.get('visualization_output_dir', 'data/08_reporting/visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    X = _extract_features(train_data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    generated_files = {}
    
    # 1. Gráfico del método del codo (K-Means)
    if 'elbow_method' in kmeans_result.get('metrics', {}):
        elbow_data = kmeans_result['metrics']['elbow_method']
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(elbow_data['k_values'], elbow_data['inertias'], marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Número de Clusters (k)', fontsize=12)
        ax.set_ylabel('Inercia (Within-cluster Sum of Squares)', fontsize=12)
        ax.set_title('Método del Codo - K-Means', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        elbow_path = os.path.join(output_dir, 'kmeans_elbow_method.png')
        plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files['kmeans_elbow'] = elbow_path
        logger.info(f"Grafíco del método del codo guardado en: {elbow_path}")
    
    # 2. Dendrograma (Hierarchical Clustering)
    if 'dendrogram_data' in hierarchical_result.get('metrics', {}):
        dendro_data = hierarchical_result['metrics']['dendrogram_data']
        linkage_matrix = np.array(dendro_data['linkage_matrix'])
        
        # Para datasets grandes, mostrar solo los últimos merges
        max_display = min(50, len(linkage_matrix) + 1)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=max_display,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax
        )
        ax.set_xlabel('Muestras o (Cluster Size)', fontsize=12)
        ax.set_ylabel('Distancia', fontsize=12)
        ax.set_title('Dendrograma - Hierarchical Clustering', fontsize=14, fontweight='bold')
        plt.tight_layout()
        dendro_path = os.path.join(output_dir, 'hierarchical_dendrogram.png')
        plt.savefig(dendro_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files['hierarchical_dendrogram'] = dendro_path
        logger.info(f"Dendrograma guardado en: {dendro_path}")
    
    # 3. Visualización de clusters usando PCA (2D)
    # Reducir a 2D para visualización
    pca_vis = PCA(n_components=2, random_state=parameters.get('random_state', 42))
    X_2d = pca_vis.fit_transform(X_scaled_df)
    
    # K-Means clusters
    kmeans_labels = kmeans_result.get('labels', [])
    if len(kmeans_labels) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='viridis', 
                           s=20, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.set_xlabel(f'Primer Componente Principal (Varianza: {pca_vis.explained_variance_ratio_[0]:.2%})', fontsize=11)
        ax.set_ylabel(f'Segundo Componente Principal (Varianza: {pca_vis.explained_variance_ratio_[1]:.2%})', fontsize=11)
        ax.set_title('Visualización de Clusters - K-Means (PCA 2D)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        kmeans_vis_path = os.path.join(output_dir, 'kmeans_clusters_visualization.png')
        plt.savefig(kmeans_vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files['kmeans_visualization'] = kmeans_vis_path
        logger.info(f"Visualización K-Means guardada en: {kmeans_vis_path}")
    
    # OPTICS clusters
    optics_labels = optics_result.get('labels', [])
    if len(optics_labels) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        # Manejar puntos de ruido (label = -1)
        unique_labels = set(optics_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=optics_labels, cmap='tab20', 
                           s=20, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.set_xlabel(f'Primer Componente Principal (Varianza: {pca_vis.explained_variance_ratio_[0]:.2%})', fontsize=11)
        ax.set_ylabel(f'Segundo Componente Principal (Varianza: {pca_vis.explained_variance_ratio_[1]:.2%})', fontsize=11)
        ax.set_title(f'Visualización de Clusters - OPTICS (PCA 2D)\n{n_clusters} clusters detectados', 
                    fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        optics_vis_path = os.path.join(output_dir, 'optics_clusters_visualization.png')
        plt.savefig(optics_vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files['optics_visualization'] = optics_vis_path
        logger.info(f"Visualización OPTICS guardada en: {optics_vis_path}")
    
    # Hierarchical clusters
    hierarchical_labels = hierarchical_result.get('labels', [])
    if len(hierarchical_labels) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=hierarchical_labels, cmap='Set3', 
                           s=20, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.set_xlabel(f'Primer Componente Principal (Varianza: {pca_vis.explained_variance_ratio_[0]:.2%})', fontsize=11)
        ax.set_ylabel(f'Segundo Componente Principal (Varianza: {pca_vis.explained_variance_ratio_[1]:.2%})', fontsize=11)
        ax.set_title('Visualización de Clusters - Hierarchical Clustering (PCA 2D)', 
                    fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        hierarchical_vis_path = os.path.join(output_dir, 'hierarchical_clusters_visualization.png')
        plt.savefig(hierarchical_vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files['hierarchical_visualization'] = hierarchical_vis_path
        logger.info(f"Visualización Hierarchical guardada en: {hierarchical_vis_path}")
    
    # 4. Comparación de métricas
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = ['K-Means', 'OPTICS', 'Hierarchical']
    silhouette_scores = [
        kmeans_result['metrics']['silhouette_score'],
        optics_result['metrics']['silhouette_score'],
        hierarchical_result['metrics']['silhouette_score']
    ]
    davies_bouldin = [
        kmeans_result['metrics']['davies_bouldin_index'],
        optics_result['metrics']['davies_bouldin_index'],
        hierarchical_result['metrics']['davies_bouldin_index']
    ]
    calinski_harabasz = [
        kmeans_result['metrics']['calinski_harabasz_index'],
        optics_result['metrics']['calinski_harabasz_index'],
        hierarchical_result['metrics']['calinski_harabasz_index']
    ]
    
    axes[0].bar(models, silhouette_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_ylabel('Silhouette Score', fontsize=11)
    axes[0].set_title('Silhouette Score (mayor es mejor)', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(models, davies_bouldin, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1].set_ylabel('Davies-Bouldin Index', fontsize=11)
    axes[1].set_title('Davies-Bouldin Index (menor es mejor)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[2].bar(models, calinski_harabasz, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[2].set_ylabel('Calinski-Harabasz Index', fontsize=11)
    axes[2].set_title('Calinski-Harabasz Index (mayor es mejor)', fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    metrics_comparison_path = os.path.join(output_dir, 'clustering_metrics_comparison.png')
    plt.savefig(metrics_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files['metrics_comparison'] = metrics_comparison_path
    logger.info(f"Comparación de métricas guardada en: {metrics_comparison_path}")
    
    logger.info(f"Total de visualizaciones generadas: {len(generated_files)}")
    
    return generated_files

