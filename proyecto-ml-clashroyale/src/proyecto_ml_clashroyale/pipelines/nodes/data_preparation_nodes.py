"""Nodos para la preparación de datos (Fase 3 CRISP-DM)."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


def select_relevant_columns(combates1_cleaned: pd.DataFrame,
                          combates2_cleaned: pd.DataFrame,
                          combates3_cleaned: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Seleccionar solo las columnas relevantes de los 3 datasets de combates.
    
    Args:
        combates1_cleaned: Dataset de combates 1 limpio
        combates2_cleaned: Dataset de combates 2 limpio
        combates3_cleaned: Dataset de combates 3 limpio
        
    Returns:
        Diccionario con los 3 datasets con columnas seleccionadas
    """
    logger.info("Seleccionando columnas relevantes de los datasets...")
    
    # Definir las columnas que necesitamos
    relevant_columns = [
        'battle_id',
        'winner.tag',
        'loser.tag',
        'winner.card1.id', 'winner.card2.id', 'winner.card3.id', 'winner.card4.id',
        'winner.card5.id', 'winner.card6.id', 'winner.card7.id', 'winner.card8.id',
        'winner.cards.list',
        'winner.common.count', 'winner.rare.count', 'winner.epic.count', 'winner.legendary.count',
        'loser.card1.id', 'loser.card2.id', 'loser.card3.id', 'loser.card4.id',
        'loser.card5.id', 'loser.card6.id', 'loser.card7.id', 'loser.card8.id',
        'loser.cards.list',
        'loser.common.count', 'loser.rare.count', 'loser.epic.count', 'loser.legendary.count'
    ]
    
    # Verificar que las columnas existen en todos los datasets
    datasets = {
        'combates1': combates1_cleaned,
        'combates2': combates2_cleaned,
        'combates3': combates3_cleaned
    }
    
    selected_datasets = {}
    
    for name, dataset in datasets.items():
        # Filtrar solo las columnas que existen en el dataset
        existing_columns = [col for col in relevant_columns if col in dataset.columns]
        
        if len(existing_columns) != len(relevant_columns):
            missing_columns = [col for col in relevant_columns if col not in dataset.columns]
            logger.warning(f"Dataset {name} no tiene las columnas: {missing_columns}")
        
        # Seleccionar las columnas existentes
        selected_dataset = dataset[existing_columns].copy()
        selected_datasets[name] = selected_dataset
        
        logger.info(f"Dataset {name}: {len(selected_dataset)} registros, {len(existing_columns)} columnas seleccionadas")
    
    return selected_datasets


def combine_datasets(selected_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combinar los 3 datasets seleccionados en uno solo.
    
    Args:
        selected_datasets: Diccionario con los 3 datasets con columnas seleccionadas
        
    Returns:
        Dataset combinado con todos los registros
    """
    logger.info("Combinando los 3 datasets...")
    
    # Combinar todos los datasets
    combined_dataset = pd.concat([
        selected_datasets['combates1'],
        selected_datasets['combates2'],
        selected_datasets['combates3']
    ], ignore_index=True)
    
    logger.info(f"Dataset combinado: {len(combined_dataset):,} registros, {len(combined_dataset.columns)} columnas")
    
    return combined_dataset


def validate_combined_dataset(combined_dataset: pd.DataFrame) -> Dict[str, Any]:
    """Validar la calidad del dataset combinado.
    
    Args:
        combined_dataset: Dataset combinado de los 3 datasets
        
    Returns:
        Diccionario con información de validación
    """
    logger.info("Validando dataset combinado...")
    
    validation_results = {
        'total_records': len(combined_dataset),
        'total_columns': len(combined_dataset.columns),
        'missing_values': combined_dataset.isnull().sum().to_dict(),
        'duplicate_records': int(combined_dataset.duplicated().sum()),
        'data_types': combined_dataset.dtypes.to_dict(),
        'memory_usage': str(combined_dataset.memory_usage(deep=True).sum() / 1024 / 1024) + ' MB'
    }
    
    # Análisis específico de las columnas de cartas
    card_columns = [col for col in combined_dataset.columns if 'card' in col.lower() and 'id' in col.lower()]
    winner_card_columns = [col for col in card_columns if 'winner' in col.lower()]
    loser_card_columns = [col for col in card_columns if 'loser' in col.lower()]
    
    validation_results['card_analysis'] = {
        'total_card_columns': len(card_columns),
        'winner_card_columns': len(winner_card_columns),
        'loser_card_columns': len(loser_card_columns),
        'winner_card_columns_names': winner_card_columns,
        'loser_card_columns_names': loser_card_columns
    }
    
    # Análisis de valores únicos en columnas clave
    validation_results['unique_values'] = {
        'unique_battle_ids': combined_dataset['battle_id'].nunique(),
        'unique_winner_tags': combined_dataset['winner.tag'].nunique(),
        'unique_loser_tags': combined_dataset['loser.tag'].nunique()
    }
    
    # Verificar integridad de datos
    validation_results['data_integrity'] = {
        'battle_ids_complete': not combined_dataset['battle_id'].isnull().any(),
        'winner_tags_complete': not combined_dataset['winner.tag'].isnull().any(),
        'loser_tags_complete': not combined_dataset['loser.tag'].isnull().any()
    }
    
    logger.info(f"Validación completada: {validation_results['total_records']:,} registros válidos")
    return validation_results


def create_preparation_summary(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Crear resumen de la preparación de datos.
    
    Args:
        validation_results: Resultados de la validación del dataset
        
    Returns:
        Diccionario con resumen de la preparación
    """
    logger.info("Creando resumen de preparación de datos...")
    
    summary = {
        'preparation_overview': {
            'phase': 'Fase 3 - Preparación de Datos',
            'total_records_combined': validation_results['total_records'],
            'total_columns_selected': validation_results['total_columns'],
            'memory_usage': validation_results['memory_usage']
        },
        'data_quality': {
            'duplicate_records': validation_results['duplicate_records'],
            'missing_values_summary': sum(validation_results['missing_values'].values()),
            'data_integrity_checks': validation_results['data_integrity']
        },
        'selected_features': {
            'card_columns_total': validation_results['card_analysis']['total_card_columns'],
            'winner_card_columns': validation_results['card_analysis']['winner_card_columns'],
            'loser_card_columns': validation_results['card_analysis']['loser_card_columns'],
            'count_columns': ['common.count', 'rare.count', 'epic.count', 'legendary.count']
        },
        'dataset_statistics': {
            'unique_battle_ids': validation_results['unique_values']['unique_battle_ids'],
            'unique_winner_tags': validation_results['unique_values']['unique_winner_tags'],
            'unique_loser_tags': validation_results['unique_values']['unique_loser_tags']
        },
        'next_steps': [
            'Dataset listo para análisis de patrones',
            'Preparado para modelado de machine learning',
            'Columnas de cartas y conteos de rareza disponibles',
            'Identificadores únicos de jugadores y batallas preservados'
        ]
    }
    
    logger.info("Resumen de preparación de datos creado exitosamente")
    return summary
