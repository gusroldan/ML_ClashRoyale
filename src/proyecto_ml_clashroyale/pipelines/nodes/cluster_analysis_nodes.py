"""Nodos para análisis profundo de clusters: estadísticas, perfiles, interpretación de negocio."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import json

logger = logging.getLogger(__name__)


def analyze_cluster_statistics(
    train_data: pd.DataFrame,
    clustering_result: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Analizar estadísticas detalladas por cluster.
    
    Args:
        train_data: Datos de entrenamiento con features
        clustering_result: Resultado del modelo de clustering (dict con 'model', 'labels', etc.)
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con estadísticas por cluster
    """
    # Extraer información del resultado
    cluster_labels = clustering_result.get('labels', None)
    if cluster_labels is None:
        # Si no hay labels, intentar predecir
        model = clustering_result.get('model')
        scaler = clustering_result.get('scaler')
        feature_cols = [col for col in train_data.columns 
                       if col not in ['label', 'target_regression', 'battle_id']]
        X = train_data[feature_cols].copy()
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        if hasattr(model, 'predict'):
            cluster_labels = model.predict(X_scaled)
        else:
            cluster_labels = model.fit_predict(X_scaled)
    
    cluster_model_name = parameters.get('model_name', 'clustering')
    
    logger.info(f"Analizando estadísticas por cluster para {cluster_model_name}...")
    
    # Extraer features (sin labels)
    feature_cols = [col for col in train_data.columns 
                   if col not in ['label', 'target_regression', 'battle_id']]
    X = train_data[feature_cols].copy()
    
    # Agregar etiquetas de cluster
    X['cluster'] = cluster_labels
    
    # Calcular estadísticas por cluster
    cluster_stats = {}
    n_clusters = len(np.unique(cluster_labels))
    
    for cluster_id in range(n_clusters):
        cluster_data = X[X['cluster'] == cluster_id]
        n_samples = len(cluster_data)
        
        if n_samples == 0:
            continue
        
        # Estadísticas básicas
        stats = {
            'cluster_id': int(cluster_id),
            'n_samples': int(n_samples),
            'percentage': float(n_samples / len(X) * 100),
            'mean_features': {},
            'std_features': {},
            'median_features': {},
            'min_features': {},
            'max_features': {}
        }
        
        # Calcular estadísticas para cada feature
        for col in feature_cols:
            if col != 'cluster':
                stats['mean_features'][col] = float(cluster_data[col].mean())
                stats['std_features'][col] = float(cluster_data[col].std())
                stats['median_features'][col] = float(cluster_data[col].median())
                stats['min_features'][col] = float(cluster_data[col].min())
                stats['max_features'][col] = float(cluster_data[col].max())
        
        # Si hay labels, calcular distribución
        if 'label' in train_data.columns:
            cluster_labels_data = train_data.loc[cluster_data.index, 'label']
            stats['win_rate'] = float(cluster_labels_data.mean())
            stats['n_wins'] = int(cluster_labels_data.sum())
            stats['n_losses'] = int(len(cluster_labels_data) - cluster_labels_data.sum())
        
        # Si hay target_regression, calcular estadísticas
        if 'target_regression' in train_data.columns:
            cluster_regression = train_data.loc[cluster_data.index, 'target_regression']
            stats['mean_trophy_change'] = float(cluster_regression.mean())
            stats['std_trophy_change'] = float(cluster_regression.std())
            stats['median_trophy_change'] = float(cluster_regression.median())
        
        cluster_stats[f'cluster_{cluster_id}'] = stats
    
    logger.info(f"Estadísticas calculadas para {n_clusters} clusters")
    
    # Agregar nombre del modelo desde parameters si está disponible
    model_name = parameters.get('model_name', cluster_model_name)
    if 'optics' in str(clustering_result).lower():
        model_name = 'OPTICS'
    elif 'kmeans' in str(clustering_result).lower():
        model_name = 'K-Means'
    elif 'hierarchical' in str(clustering_result).lower():
        model_name = 'Hierarchical'
    
    return {
        'model_name': model_name,
        'n_clusters': int(n_clusters),
        'total_samples': int(len(X)),
        'cluster_statistics': cluster_stats
    }


def create_cluster_profiles(
    cluster_stats: Dict[str, Any],
    card_master_list: pd.DataFrame
) -> Dict[str, Any]:
    """Crear perfiles interpretables de cada cluster.
    
    Args:
        cluster_stats: Estadísticas por cluster
        card_master_list: Lista maestra de cartas para interpretación
        
    Returns:
        Diccionario con perfiles de clusters
    """
    logger.info("Creando perfiles de clusters...")
    
    profiles = {}
    cluster_stats_data = cluster_stats['cluster_statistics']
    
    # Crear diccionario de nombres de cartas si está disponible
    card_names = {}
    if card_master_list is not None and 'id' in card_master_list.columns:
        if 'name' in card_master_list.columns:
            card_names = dict(zip(card_master_list['id'], card_master_list['name']))
        elif 'Name' in card_master_list.columns:
            card_names = dict(zip(card_master_list['id'], card_master_list['Name']))
    
    for cluster_key, stats in cluster_stats_data.items():
        cluster_id = stats['cluster_id']
        
        # Identificar features más distintivas (mayor diferencia respecto a la media global)
        mean_features = stats['mean_features']
        
        # Ordenar features por valor absoluto de la media
        sorted_features = sorted(
            mean_features.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Top 10 features más distintivas
        top_features = {
            feature: {
                'mean_value': float(value),
                'interpretation': _interpret_feature(feature, value, card_names)
            }
            for feature, value in sorted_features[:10]
        }
        
        # Crear perfil del cluster
        profile = {
            'cluster_id': cluster_id,
            'size': stats['n_samples'],
            'percentage': stats['percentage'],
            'characteristics': {
                'top_features': top_features,
                'win_rate': stats.get('win_rate', None),
                'mean_trophy_change': stats.get('mean_trophy_change', None)
            },
            'semantic_label': _generate_semantic_label(stats, top_features)
        }
        
        profiles[f'cluster_{cluster_id}'] = profile
    
    logger.info(f"Perfiles creados para {len(profiles)} clusters")
    
    return {
        'model_name': cluster_stats['model_name'],
        'n_clusters': cluster_stats['n_clusters'],
        'profiles': profiles
    }


def _interpret_feature(feature_name: str, value: float, card_names: Dict) -> str:
    """Interpretar el significado de un feature.
    
    Args:
        feature_name: Nombre del feature
        value: Valor del feature
        card_names: Diccionario de nombres de cartas
        
    Returns:
        Interpretación del feature
    """
    # Interpretar features de cartas
    if 'cards_diff' in feature_name or 'card_' in feature_name:
        card_id = feature_name.replace('cards_diff_', '').replace('card_', '')
        card_name = card_names.get(int(card_id), f"Carta {card_id}")
        
        if value > 0.5:
            return f"Fuerte presencia de {card_name} en mazos del jugador A"
        elif value < -0.5:
            return f"Fuerte presencia de {card_name} en mazos del jugador B"
        else:
            return f"Presencia balanceada de {card_name}"
    
    # Interpretar features de rareza
    if 'common' in feature_name.lower():
        if value > 0:
            return "Mayor cantidad de cartas comunes"
        else:
            return "Menor cantidad de cartas comunes"
    
    if 'rare' in feature_name.lower():
        if value > 0:
            return "Mayor cantidad de cartas raras"
        else:
            return "Menor cantidad de cartas raras"
    
    if 'epic' in feature_name.lower():
        if value > 0:
            return "Mayor cantidad de cartas épicas"
        else:
            return "Menor cantidad de cartas épicas"
    
    if 'legendary' in feature_name.lower():
        if value > 0:
            return "Mayor cantidad de cartas legendarias"
        else:
            return "Menor cantidad de cartas legendarias"
    
    # Interpretar diferencia de trofeos
    if 'trophies' in feature_name.lower() or 'delta' in feature_name.lower():
        if value > 0:
            return "Jugador A tiene más trofeos iniciales"
        else:
            return "Jugador B tiene más trofeos iniciales"
    
    return f"Feature {feature_name} con valor {value:.2f}"


def _generate_semantic_label(stats: Dict[str, Any], top_features: Dict[str, Any]) -> str:
    """Generar etiqueta semántica para el cluster.
    
    Args:
        stats: Estadísticas del cluster
        top_features: Top features distintivas
        
    Returns:
        Etiqueta semántica del cluster
    """
    # Analizar características principales
    characteristics = []
    
    # Analizar win rate
    win_rate = stats.get('win_rate', None)
    if win_rate is not None:
        if win_rate > 0.6:
            characteristics.append("Alto Win Rate")
        elif win_rate < 0.4:
            characteristics.append("Bajo Win Rate")
        else:
            characteristics.append("Win Rate Balanceado")
    
    # Analizar cambio de trofeos
    trophy_change = stats.get('mean_trophy_change', None)
    if trophy_change is not None:
        if trophy_change > 20:
            characteristics.append("Ganadores de Trofeos")
        elif trophy_change < -20:
            characteristics.append("Perdedores de Trofeos")
    
    # Analizar features de cartas
    card_features = [f for f in top_features.keys() if 'card' in f.lower() or 'cards' in f.lower()]
    if len(card_features) > 5:
        characteristics.append("Mazos Diversos")
    else:
        characteristics.append("Mazos Especializados")
    
    # Analizar rareza
    rarity_features = [f for f in top_features.keys() if any(r in f.lower() for r in ['common', 'rare', 'epic', 'legendary'])]
    if any('legendary' in f.lower() for f in rarity_features):
        characteristics.append("Cartas Legendarias")
    elif any('epic' in f.lower() for f in rarity_features):
        characteristics.append("Cartas Épicas")
    
    # Generar etiqueta
    if characteristics:
        label = " | ".join(characteristics[:3])  # Máximo 3 características
    else:
        label = f"Cluster {stats['cluster_id']}"
    
    return label


def create_business_interpretation(
    cluster_profiles: Dict[str, Any]
) -> Dict[str, Any]:
    """Crear interpretación de negocio de los clusters.
    
    Args:
        cluster_profiles: Perfiles de clusters
        
    Returns:
        Diccionario con interpretación de negocio
    """
    logger.info("Creando interpretación de negocio...")
    
    profiles = cluster_profiles['profiles']
    
    # Agrupar clusters por características similares
    high_win_rate_clusters = []
    low_win_rate_clusters = []
    balanced_clusters = []
    
    for cluster_key, profile in profiles.items():
        win_rate = profile['characteristics'].get('win_rate', None)
        if win_rate is not None:
            if win_rate > 0.6:
                high_win_rate_clusters.append(profile)
            elif win_rate < 0.4:
                low_win_rate_clusters.append(profile)
            else:
                balanced_clusters.append(profile)
    
    # Crear insights de negocio
    insights = {
        'high_performance_clusters': {
            'count': len(high_win_rate_clusters),
            'description': 'Clusters con win rate superior al 60%',
            'recommendations': [
                'Analizar estrategias de estos clusters para replicar en otros',
                'Identificar cartas y combinaciones más efectivas',
                'Usar como referencia para recomendaciones de mazos'
            ]
        },
        'low_performance_clusters': {
            'count': len(low_win_rate_clusters),
            'description': 'Clusters con win rate inferior al 40%',
            'recommendations': [
                'Identificar debilidades en estos clusters',
                'Evitar combinaciones de cartas que caracterizan estos clusters',
                'Mejorar estrategias para estos tipos de mazos'
            ]
        },
        'balanced_clusters': {
            'count': len(balanced_clusters),
            'description': 'Clusters con win rate balanceado (40-60%)',
            'recommendations': [
                'Estos clusters representan el meta promedio',
                'Pueden beneficiarse de pequeñas mejoras',
                'Son buenos para análisis comparativo'
            ]
        }
    }
    
    # Análisis de tamaño de clusters
    cluster_sizes = [profile['size'] for profile in profiles.values()]
    insights['cluster_size_analysis'] = {
        'largest_cluster': max(cluster_sizes),
        'smallest_cluster': min(cluster_sizes),
        'average_size': float(np.mean(cluster_sizes)),
        'interpretation': 'Clusters grandes indican estrategias comunes, clusters pequeños indican estrategias especializadas'
    }
    
    logger.info("Interpretación de negocio completada")
    
    return {
        'model_name': cluster_profiles['model_name'],
        'n_clusters': cluster_profiles['n_clusters'],
        'business_insights': insights,
        'cluster_profiles': profiles
    }

