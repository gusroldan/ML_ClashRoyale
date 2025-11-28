"""Nodos para el pipeline de reglas de asociación."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
import json

logger = logging.getLogger(__name__)

# Intentar importar mlxtend para reglas de asociación
try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    logger.warning("mlxtend no está disponible. Las reglas de asociación no se podrán usar.")


def prepare_transaction_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preparar datos en formato de transacciones para reglas de asociación.
    
    Convierte las features de cartas (multi-hot encoding) en formato de transacciones.
    Cada fila es una transacción (batalla) y cada columna es un ítem (carta).
    
    Args:
        data: DataFrame con features (debe incluir columnas de cartas)
        
    Returns:
        DataFrame binario con formato de transacciones
    """
    # Extraer solo las columnas de cartas (asumiendo que tienen nombres como 'card_X' o similar)
    # O usar todas las columnas numéricas binarias
    card_columns = [col for col in data.columns if 'card' in col.lower() or col.startswith('card_')]
    
    if len(card_columns) == 0:
        # Si no hay columnas de cartas, usar todas las columnas numéricas binarias
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        # Eliminar columnas de labels
        numeric_cols = [col for col in numeric_cols if col not in ['label', 'target_regression']]
        card_columns = numeric_cols[:50]  # Limitar a 50 features para evitar problemas de memoria
        logger.warning(f"No se encontraron columnas de cartas. Usando {len(card_columns)} features numéricas.")
    
    # Crear DataFrame de transacciones (binario)
    transaction_data = data[card_columns].copy()
    
    # Convertir a binario (valores > 0 = True, 0 = False)
    transaction_data = (transaction_data > 0).astype(int)
    
    # Renombrar columnas para que sean más legibles
    transaction_data.columns = [f"Item_{i}" for i in range(len(transaction_data.columns))]
    
    return transaction_data


def save_association_rules_metrics(rules: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    """Guardar métricas de reglas de asociación en formato serializable.
    
    Args:
        rules: DataFrame con reglas de asociación
        model_name: Nombre del algoritmo
        
    Returns:
        Diccionario con métricas convertidas a tipos básicos
    """
    if rules.empty:
        return {
            'model_name': model_name,
            'n_rules': 0,
            'message': 'No se encontraron reglas con los parámetros especificados'
        }
    
    # Obtener estadísticas de las reglas
    result = {
        'model_name': model_name,
        'n_rules': int(len(rules)),
        'metrics_summary': {
            'support': {
                'mean': float(rules['support'].mean()),
                'min': float(rules['support'].min()),
                'max': float(rules['support'].max())
            },
            'confidence': {
                'mean': float(rules['confidence'].mean()),
                'min': float(rules['confidence'].min()),
                'max': float(rules['confidence'].max())
            },
            'lift': {
                'mean': float(rules['lift'].mean()),
                'min': float(rules['lift'].min()),
                'max': float(rules['lift'].max())
            }
        },
        'top_rules': []
    }
    
    # Guardar top 10 reglas por lift
    top_rules = rules.nlargest(10, 'lift')
    for idx, rule in top_rules.iterrows():
        result['top_rules'].append({
            'antecedents': str(rule['antecedents']),
            'consequents': str(rule['consequents']),
            'support': float(rule['support']),
            'confidence': float(rule['confidence']),
            'lift': float(rule['lift'])
        })
    
    return result


def find_association_rules_apriori(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Encontrar reglas de asociación usando el algoritmo Apriori.
    
    Args:
        train_data: Data de entrenamiento
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (resultado del modelo, métricas en formato JSON)
    """
    if not MLXTEND_AVAILABLE:
        logger.error("mlxtend no está disponible. No se puede usar Apriori.")
        raise ImportError("mlxtend es requerido para reglas de asociación. Instala con: pip install mlxtend")
    
    logger.info("Encontrando reglas de asociación con algoritmo Apriori...")
    
    # Preparar datos en formato de transacciones
    transaction_data = prepare_transaction_data(train_data)
    
    # Parámetros
    min_support = parameters['models']['apriori'].get('min_support', 0.1)
    use_colnames = parameters['models']['apriori'].get('use_colnames', True)
    max_len = parameters['models']['apriori'].get('max_len', None)
    
    # Encontrar itemsets frecuentes
    frequent_itemsets = apriori(
        transaction_data,
        min_support=min_support,
        use_colnames=use_colnames,
        max_len=max_len,
        verbose=0
    )
    
    if frequent_itemsets.empty:
        logger.warning("No se encontraron itemsets frecuentes con los parámetros especificados.")
        metrics_json = {
            'model_name': 'Apriori',
            'n_rules': 0,
            'message': 'No se encontraron itemsets frecuentes'
        }
        return {
            'frequent_itemsets': pd.DataFrame(),
            'rules': pd.DataFrame(),
            'metrics': {}
        }, metrics_json
    
    # Generar reglas de asociación
    min_confidence = parameters['models']['apriori'].get('min_confidence', 0.5)
    min_lift = parameters['models']['apriori'].get('min_lift', 1.0)
    
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
    
    # Filtrar por lift mínimo
    if not rules.empty:
        rules = rules[rules['lift'] >= min_lift]
    
    metrics = {
        'n_itemsets': int(len(frequent_itemsets)),
        'n_rules': int(len(rules)),
        'parameters': {
            'min_support': min_support,
            'min_confidence': min_confidence,
            'min_lift': min_lift,
            'max_len': max_len
        }
    }
    
    logger.info(f"Apriori - Itemsets frecuentes: {len(frequent_itemsets)}, Reglas: {len(rules)}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_association_rules_metrics(rules, 'Apriori')
    metrics_json['n_itemsets'] = int(len(frequent_itemsets))
    
    return {
        'frequent_itemsets': frequent_itemsets,
        'rules': rules,
        'metrics': metrics
    }, metrics_json


def find_association_rules_fpgrowth(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Encontrar reglas de asociación usando el algoritmo FP-Growth.
    
    Args:
        train_data: Data de entrenamiento
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (resultado del modelo, métricas en formato JSON)
    """
    if not MLXTEND_AVAILABLE:
        logger.error("mlxtend no está disponible. No se puede usar FP-Growth.")
        raise ImportError("mlxtend es requerido para reglas de asociación. Instala con: pip install mlxtend")
    
    logger.info("Encontrando reglas de asociación con algoritmo FP-Growth...")
    
    # Preparar datos en formato de transacciones
    transaction_data = prepare_transaction_data(train_data)
    
    # Parámetros
    min_support = parameters['models']['fpgrowth'].get('min_support', 0.1)
    use_colnames = parameters['models']['fpgrowth'].get('use_colnames', True)
    max_len = parameters['models']['fpgrowth'].get('max_len', None)
    
    # Encontrar itemsets frecuentes
    frequent_itemsets = fpgrowth(
        transaction_data,
        min_support=min_support,
        use_colnames=use_colnames,
        max_len=max_len,
        verbose=0
    )
    
    if frequent_itemsets.empty:
        logger.warning("No se encontraron itemsets frecuentes con los parámetros especificados.")
        metrics_json = {
            'model_name': 'FP-Growth',
            'n_rules': 0,
            'message': 'No se encontraron itemsets frecuentes'
        }
        return {
            'frequent_itemsets': pd.DataFrame(),
            'rules': pd.DataFrame(),
            'metrics': {}
        }, metrics_json
    
    # Generar reglas de asociación
    min_confidence = parameters['models']['fpgrowth'].get('min_confidence', 0.5)
    min_lift = parameters['models']['fpgrowth'].get('min_lift', 1.0)
    
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
    
    # Filtrar por lift mínimo
    if not rules.empty:
        rules = rules[rules['lift'] >= min_lift]
    
    metrics = {
        'n_itemsets': int(len(frequent_itemsets)),
        'n_rules': int(len(rules)),
        'parameters': {
            'min_support': min_support,
            'min_confidence': min_confidence,
            'min_lift': min_lift,
            'max_len': max_len
        }
    }
    
    logger.info(f"FP-Growth - Itemsets frecuentes: {len(frequent_itemsets)}, Reglas: {len(rules)}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_association_rules_metrics(rules, 'FP-Growth')
    metrics_json['n_itemsets'] = int(len(frequent_itemsets))
    
    return {
        'frequent_itemsets': frequent_itemsets,
        'rules': rules,
        'metrics': metrics
    }, metrics_json


def create_association_rules_comparison(
    apriori_result: Dict[str, Any],
    fpgrowth_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Crear comparación entre Apriori y FP-Growth.
    
    Args:
        apriori_result: Resultado de Apriori
        fpgrowth_result: Resultado de FP-Growth
        
    Returns:
        Diccionario con comparación de algoritmos
    """
    logger.info("Creando comparación de algoritmos de reglas de asociación...")
    
    comparison = {
        'apriori': {
            'n_itemsets': int(apriori_result['metrics'].get('n_itemsets', 0)),
            'n_rules': int(apriori_result['metrics'].get('n_rules', 0))
        },
        'fpgrowth': {
            'n_itemsets': int(fpgrowth_result['metrics'].get('n_itemsets', 0)),
            'n_rules': int(fpgrowth_result['metrics'].get('n_rules', 0))
        }
    }
    
    # Comparar métricas si hay reglas
    if not apriori_result['rules'].empty and not fpgrowth_result['rules'].empty:
        comparison['apriori']['avg_lift'] = float(apriori_result['rules']['lift'].mean())
        comparison['fpgrowth']['avg_lift'] = float(fpgrowth_result['rules']['lift'].mean())
        comparison['apriori']['avg_confidence'] = float(apriori_result['rules']['confidence'].mean())
        comparison['fpgrowth']['avg_confidence'] = float(fpgrowth_result['rules']['confidence'].mean())
    
    comparison['summary'] = {
        'note': 'FP-Growth es generalmente más rápido que Apriori, especialmente con datasets grandes',
        'best_for': {
            'small_datasets': 'Ambos algoritmos funcionan bien',
            'large_datasets': 'FP-Growth (más eficiente)',
            'memory_constraints': 'Apriori (usa menos memoria)'
        }
    }
    
    logger.info("Comparación de reglas de asociación completada")
    
    return comparison

