"""Nodos para el pipeline de reducción de dimensionalidad."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import logging
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


def save_dimensionality_reduction_metrics(metrics: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Guardar métricas de reducción de dimensionalidad en formato serializable.
    
    Args:
        metrics: Diccionario con métricas
        model_name: Nombre del modelo
        
    Returns:
        Diccionario con métricas convertidas a tipos básicos
    """
    result = {
        'model_name': model_name,
        'n_components': int(metrics.get('n_components', 0)),
        'original_dimensions': int(metrics.get('original_dimensions', 0)),
        'parameters': metrics.get('parameters', {})
    }
    
    # Agregar métricas específicas si existen
    if 'explained_variance' in metrics:
        result['explained_variance'] = {
            'per_component': [float(v) for v in metrics['explained_variance']['per_component']],
            'cumulative': [float(v) for v in metrics['explained_variance']['cumulative']],
            'total_explained_variance': float(metrics['explained_variance']['total_explained_variance'])
        }
    
    if 'loadings' in metrics:
        # Guardar solo los primeros componentes para no hacer el JSON muy grande
        # Los loadings completos se guardan en el modelo pickle
        n_components_to_save = min(10, len(metrics['loadings']['component_names']))
        result['loadings'] = {
            'component_names': metrics['loadings']['component_names'][:n_components_to_save],
            'feature_names': metrics['loadings']['feature_names'],
            'top_features_per_component': metrics['loadings'].get('top_features_per_component', {})
        }
    
    if 'biplot_data' in metrics:
        result['biplot_data'] = {
            'n_samples': int(metrics['biplot_data']['n_samples']),
            'n_components': int(metrics['biplot_data']['n_components']),
            'note': 'Datos completos del biplot disponibles en el modelo pickle'
        }
    
    return result


def apply_pca(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Aplicar PCA (Análisis de Componentes Principales).
    
    Args:
        train_data: Data de entrenamiento
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (resultado del modelo, métricas en formato JSON)
    """
    logger.info("Aplicando PCA (Análisis de Componentes Principales)...")
    
    X = _extract_features(train_data)
    original_dimensions = X.shape[1]
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Parámetros
    n_components = parameters['models']['pca'].get('n_components', None)
    if n_components is None or n_components == 'auto':
        # Si no se especifica, usar el 95% de varianza explicada
        n_components = parameters['models']['pca'].get('variance_threshold', 0.95)
    
    # Aplicar PCA
    if isinstance(n_components, float) and 0 < n_components < 1:
        # Si es un float, interpretar como fracción de varianza
        pca = PCA(n_components=n_components, random_state=parameters.get('random_state', 42))
    else:
        # Si es un int, usar ese número de componentes
        pca = PCA(n_components=int(n_components), random_state=parameters.get('random_state', 42))
    
    X_transformed = pca.fit_transform(X_scaled_df)
    
    # Calcular varianza explicada
    explained_variance_per_component = pca.explained_variance_ratio_
    explained_variance_cumulative = np.cumsum(explained_variance_per_component)
    total_explained_variance = float(explained_variance_cumulative[-1])
    n_components_actual = len(explained_variance_per_component)
    
    # Calcular loadings (componentes principales)
    loadings = pca.components_.T  # Transponer para tener features x componentes
    loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(n_components_actual)], index=X.columns)
    
    # Obtener top features por componente (mayores valores absolutos)
    top_features_per_component = {}
    n_top_features = parameters['models']['pca'].get('n_top_features_per_component', 5)
    
    for i in range(min(n_components_actual, 10)):  # Solo primeros 10 componentes
        component_name = f'PC{i+1}'
        component_loadings = loadings_df[component_name].abs().sort_values(ascending=False)
        top_features = component_loadings.head(n_top_features).to_dict()
        top_features_per_component[component_name] = {
            k: float(v) for k, v in top_features.items()
        }
    
    # Preparar datos para biplot (solo para visualización, guardar resumen)
    # El biplot completo se puede reconstruir desde el modelo y los datos transformados
    biplot_data = {
        'n_samples': int(X_scaled_df.shape[0]),
        'n_components': int(n_components_actual),
        'note': 'Datos completos del biplot disponibles en el modelo pickle'
    }
    
    metrics = {
        'n_components': n_components_actual,
        'original_dimensions': original_dimensions,
        'explained_variance': {
            'per_component': explained_variance_per_component.tolist(),
            'cumulative': explained_variance_cumulative.tolist(),
            'total_explained_variance': total_explained_variance
        },
        'loadings': {
            'component_names': [f'PC{i+1}' for i in range(n_components_actual)],
            'feature_names': X.columns.tolist(),
            'top_features_per_component': top_features_per_component
        },
        'biplot_data': biplot_data,
        'parameters': {
            'n_components': int(n_components_actual) if isinstance(n_components_actual, (int, np.integer)) else float(n_components_actual),
            'variance_threshold': parameters['models']['pca'].get('variance_threshold', None),
            'random_state': parameters.get('random_state', 42)
        }
    }
    
    logger.info(f"PCA - Componentes: {n_components_actual}, Varianza explicada total: {total_explained_variance:.4f} "
                f"({total_explained_variance*100:.2f}%)")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_dimensionality_reduction_metrics(metrics, 'PCA')
    
    return {
        'model': pca,
        'scaler': scaler,
        'transformed_data': X_transformed,
        'loadings_df': loadings_df,
        'metrics': metrics
    }, metrics_json


def apply_umap(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Aplicar UMAP (Uniform Manifold Approximation and Projection).
    
    Args:
        train_data: Data de entrenamiento
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (resultado del modelo, métricas en formato JSON)
    """
    logger.info("Aplicando UMAP (Uniform Manifold Approximation and Projection)...")
    
    X = _extract_features(train_data)
    original_dimensions = X.shape[1]
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Parámetros
    n_components = parameters['models']['umap'].get('n_components', 2)
    n_neighbors = parameters['models']['umap'].get('n_neighbors', 15)
    min_dist = parameters['models']['umap'].get('min_dist', 0.1)
    metric = parameters['models']['umap'].get('metric', 'euclidean')
    random_state = parameters.get('random_state', 42)
    
    # Aplicar UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False
    )
    
    X_transformed = reducer.fit_transform(X_scaled_df)
    
    metrics = {
        'n_components': int(n_components),
        'original_dimensions': original_dimensions,
        'parameters': {
            'n_components': int(n_components),
            'n_neighbors': int(n_neighbors),
            'min_dist': float(min_dist),
            'metric': metric,
            'random_state': int(random_state)
        }
    }
    
    logger.info(f"UMAP - Componentes: {n_components}, Dimensiones originales: {original_dimensions} "
                f"-> {n_components} dimensiones")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_dimensionality_reduction_metrics(metrics, 'UMAP')
    
    return {
        'model': reducer,
        'scaler': scaler,
        'transformed_data': X_transformed,
        'metrics': metrics
    }, metrics_json


def create_dimensionality_reduction_comparison(
    pca_result: Dict[str, Any],
    umap_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Crear comparación entre PCA y UMAP.
    
    Args:
        pca_result: Resultado de PCA
        umap_result: Resultado de UMAP
        
    Returns:
        Diccionario con comparación de métodos
    """
    logger.info("Creando comparación de métodos de reducción de dimensionalidad...")
    
    comparison = {
        'pca': {
            'n_components': int(pca_result['metrics']['n_components']),
            'original_dimensions': int(pca_result['metrics']['original_dimensions']),
            'reduction_ratio': float(pca_result['metrics']['n_components'] / pca_result['metrics']['original_dimensions']),
            'total_explained_variance': float(pca_result['metrics']['explained_variance']['total_explained_variance']),
            'method': 'Linear transformation based on variance'
        },
        'umap': {
            'n_components': int(umap_result['metrics']['n_components']),
            'original_dimensions': int(umap_result['metrics']['original_dimensions']),
            'reduction_ratio': float(umap_result['metrics']['n_components'] / umap_result['metrics']['original_dimensions']),
            'method': 'Non-linear manifold learning'
        }
    }
    
    # Agregar información sobre cuál método reduce más
    pca_reduction = comparison['pca']['reduction_ratio']
    umap_reduction = comparison['umap']['reduction_ratio']
    
    comparison['summary'] = {
        'pca': {
            'variance_explained_percentage': float(comparison['pca']['total_explained_variance'] * 100),
            'compression_ratio': float(1 / pca_reduction) if pca_reduction > 0 else 0
        },
        'umap': {
            'compression_ratio': float(1 / umap_reduction) if umap_reduction > 0 else 0
        },
        'best_for': {
            'linear_data': 'PCA',
            'non_linear_manifolds': 'UMAP',
            'interpretability': 'PCA (loadings available)',
            'visualization': 'UMAP (better preserves local structure)'
        }
    }
    
    logger.info("Comparación de reducción de dimensionalidad completada")
    
    return comparison


def generate_dimensionality_reduction_visualizations(
    train_data: pd.DataFrame,
    pca_result: Dict[str, Any],
    umap_result: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, str]:
    """Generar y guardar visualizaciones de reducción de dimensionalidad en formato PNG.
    
    Args:
        train_data: Datos de entrenamiento
        pca_result: Resultado de PCA
        umap_result: Resultado de UMAP
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con las rutas de los archivos PNG generados
    """
    logger.info("Generando visualizaciones de reducción de dimensionalidad...")
    
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
    generated_files = {}
    
    # 1. Gráfico de varianza explicada acumulada (PCA)
    if 'explained_variance' in pca_result.get('metrics', {}):
        explained_var = pca_result['metrics']['explained_variance']
        cumulative_var = explained_var['cumulative']
        per_component_var = explained_var['per_component']
        n_components = len(cumulative_var)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico de varianza explicada por componente (Scree Plot)
        axes[0].bar(range(1, min(n_components + 1, 21)), per_component_var[:20], 
                   color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Componente Principal', fontsize=12)
        axes[0].set_ylabel('Varianza Explicada', fontsize=12)
        axes[0].set_title('Scree Plot - Varianza Explicada por Componente (PCA)', 
                         fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        if n_components > 20:
            axes[0].set_xlim(0, 21)
        
        # Gráfico de varianza acumulada
        axes[1].plot(range(1, n_components + 1), cumulative_var, 
                    marker='o', linewidth=2, markersize=6, color='coral')
        axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Varianza', alpha=0.7)
        axes[1].axhline(y=0.90, color='orange', linestyle='--', label='90% Varianza', alpha=0.7)
        axes[1].set_xlabel('Número de Componentes', fontsize=12)
        axes[1].set_ylabel('Varianza Explicada Acumulada', fontsize=12)
        axes[1].set_title('Varianza Explicada Acumulada (PCA)', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)
        
        plt.tight_layout()
        pca_variance_path = os.path.join(output_dir, 'pca_variance_explained.png')
        plt.savefig(pca_variance_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files['pca_variance'] = pca_variance_path
        logger.info(f"Gráfico de varianza PCA guardado en: {pca_variance_path}")
    
    # 2. Biplot de PCA (primeros 2 componentes)
    if 'transformed_data' in pca_result:
        X_pca = pca_result['transformed_data']
        if X_pca.shape[1] >= 2:
            # Submuestrear si hay muchos puntos para mejorar la visualización
            n_samples_vis = min(5000, len(X_pca))
            if len(X_pca) > n_samples_vis:
                indices = np.random.choice(len(X_pca), n_samples_vis, replace=False)
                X_pca_vis = X_pca[indices]
            else:
                X_pca_vis = X_pca
            
            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = ax.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], 
                               s=15, alpha=0.5, c=range(len(X_pca_vis)), 
                               cmap='viridis', edgecolors='k', linewidth=0.3)
            ax.set_xlabel(f'Primer Componente Principal (PC1)\nVarianza: {pca_result["metrics"]["explained_variance"]["per_component"][0]:.2%}', 
                         fontsize=11)
            ax.set_ylabel(f'Segundo Componente Principal (PC2)\nVarianza: {pca_result["metrics"]["explained_variance"]["per_component"][1]:.2%}', 
                         fontsize=11)
            ax.set_title('Visualización PCA - Primeros 2 Componentes Principales', 
                        fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Índice de Muestra')
            plt.tight_layout()
            pca_biplot_path = os.path.join(output_dir, 'pca_biplot_2d.png')
            plt.savefig(pca_biplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files['pca_biplot'] = pca_biplot_path
            logger.info(f"Biplot PCA guardado en: {pca_biplot_path}")
    
    # 3. Visualización UMAP (2D)
    if 'transformed_data' in umap_result:
        X_umap = umap_result['transformed_data']
        if X_umap.shape[1] >= 2:
            # Submuestrear si hay muchos puntos
            n_samples_vis = min(5000, len(X_umap))
            if len(X_umap) > n_samples_vis:
                indices = np.random.choice(len(X_umap), n_samples_vis, replace=False)
                X_umap_vis = X_umap[indices]
            else:
                X_umap_vis = X_umap
            
            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = ax.scatter(X_umap_vis[:, 0], X_umap_vis[:, 1], 
                               s=15, alpha=0.5, c=range(len(X_umap_vis)), 
                               cmap='plasma', edgecolors='k', linewidth=0.3)
            ax.set_xlabel('Primera Dimensión UMAP', fontsize=12)
            ax.set_ylabel('Segunda Dimensión UMAP', fontsize=12)
            ax.set_title('Visualización UMAP - Proyección 2D', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Índice de Muestra')
            plt.tight_layout()
            umap_vis_path = os.path.join(output_dir, 'umap_visualization_2d.png')
            plt.savefig(umap_vis_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files['umap_visualization'] = umap_vis_path
            logger.info(f"Visualización UMAP guardada en: {umap_vis_path}")
    
    # 4. Comparación PCA vs UMAP (lado a lado)
    if 'transformed_data' in pca_result and 'transformed_data' in umap_result:
        X_pca = pca_result['transformed_data']
        X_umap = umap_result['transformed_data']
        
        if X_pca.shape[1] >= 2 and X_umap.shape[1] >= 2:
            n_samples_vis = min(3000, len(X_pca))
            if len(X_pca) > n_samples_vis:
                indices = np.random.choice(len(X_pca), n_samples_vis, replace=False)
                X_pca_vis = X_pca[indices]
                X_umap_vis = X_umap[indices]
            else:
                X_pca_vis = X_pca
                X_umap_vis = X_umap
            
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            
            # PCA
            scatter1 = axes[0].scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], 
                                      s=15, alpha=0.5, c=range(len(X_pca_vis)), 
                                      cmap='viridis', edgecolors='k', linewidth=0.2)
            axes[0].set_xlabel(f'PC1 (Varianza: {pca_result["metrics"]["explained_variance"]["per_component"][0]:.2%})', 
                              fontsize=11)
            axes[0].set_ylabel(f'PC2 (Varianza: {pca_result["metrics"]["explained_variance"]["per_component"][1]:.2%})', 
                              fontsize=11)
            axes[0].set_title('PCA - Proyección Lineal', fontsize=13, fontweight='bold')
            plt.colorbar(scatter1, ax=axes[0], label='Índice')
            
            # UMAP
            scatter2 = axes[1].scatter(X_umap_vis[:, 0], X_umap_vis[:, 1], 
                                      s=15, alpha=0.5, c=range(len(X_umap_vis)), 
                                      cmap='plasma', edgecolors='k', linewidth=0.2)
            axes[1].set_xlabel('UMAP Dimensión 1', fontsize=11)
            axes[1].set_ylabel('UMAP Dimensión 2', fontsize=11)
            axes[1].set_title('UMAP - Proyección No Lineal', fontsize=13, fontweight='bold')
            plt.colorbar(scatter2, ax=axes[1], label='Índice')
            
            plt.suptitle('Comparación: PCA vs UMAP', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, 'pca_vs_umap_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files['pca_umap_comparison'] = comparison_path
            logger.info(f"Comparación PCA vs UMAP guardada en: {comparison_path}")
    
    logger.info(f"Total de visualizaciones generadas: {len(generated_files)}")
    
    return generated_files

