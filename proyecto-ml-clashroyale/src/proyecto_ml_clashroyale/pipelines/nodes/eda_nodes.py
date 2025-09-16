"""Nodos para el análisis exploratorio de datos (EDA)."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

logger = logging.getLogger(__name__)


def analyze_rarity_distributions(combates1_cleaned: pd.DataFrame, 
                                combates2_cleaned: pd.DataFrame,
                                combates3_cleaned: pd.DataFrame) -> Dict[str, Any]:
    """Analizar la distribución de rarezas en los mazos.
    
    Args:
        combates1_cleaned: Dataset de combates 1 limpio
        combates2_cleaned: Dataset de combates 2 limpio
        combates3_cleaned: Dataset de combates 3 limpio
        
    Returns:
        Diccionario con análisis de distribución de rarezas
    """
    logger.info("Analizando distribución de rarezas en los mazos...")
    
    # Combinar todos los datasets limpios
    all_combates = pd.concat([combates1_cleaned, combates2_cleaned, combates3_cleaned], 
                            ignore_index=True)
    
    # Análisis de distribución de rarezas para ganadores
    winner_rarity_stats = {}
    winner_rarity_cols = ['winner.common.count', 'winner.rare.count', 'winner.epic.count', 'winner.legendary.count']
    
    for col in winner_rarity_cols:
        if col in all_combates.columns:
            rarity_type = col.split('.')[1]  # common, rare, epic, legendary
            winner_rarity_stats[rarity_type] = {
                'mean': float(all_combates[col].mean()),
                'median': float(all_combates[col].median()),
                'std': float(all_combates[col].std()),
                'min': int(all_combates[col].min()),
                'max': int(all_combates[col].max()),
                'distribution': all_combates[col].value_counts().head(10).to_dict()
            }
    
    # Análisis de distribución de rarezas para perdedores
    loser_rarity_stats = {}
    loser_rarity_cols = ['loser.common.count', 'loser.rare.count', 'loser.epic.count', 'loser.legendary.count']
    
    for col in loser_rarity_cols:
        if col in all_combates.columns:
            rarity_type = col.split('.')[1]  # common, rare, epic, legendary
            loser_rarity_stats[rarity_type] = {
                'mean': float(all_combates[col].mean()),
                'median': float(all_combates[col].median()),
                'std': float(all_combates[col].std()),
                'min': int(all_combates[col].min()),
                'max': int(all_combates[col].max()),
                'distribution': all_combates[col].value_counts().head(10).to_dict()
            }
    
    # Análisis comparativo ganadores vs perdedores
    comparison_stats = {}
    for rarity in ['common', 'rare', 'epic', 'legendary']:
        winner_col = f'winner.{rarity}.count'
        loser_col = f'loser.{rarity}.count'
        
        if winner_col in all_combates.columns and loser_col in all_combates.columns:
            comparison_stats[rarity] = {
                'winner_avg': float(all_combates[winner_col].mean()),
                'loser_avg': float(all_combates[loser_col].mean()),
                'difference': float(all_combates[winner_col].mean() - all_combates[loser_col].mean())
            }
    
    result = {
        'winner_rarity_distribution': winner_rarity_stats,
        'loser_rarity_distribution': loser_rarity_stats,
        'winner_vs_loser_comparison': comparison_stats,
        'total_records_analyzed': len(all_combates)
    }
    
    logger.info(f"Análisis de rarezas completado para {len(all_combates)} registros")
    return result


def analyze_most_used_cards(combates1_cleaned: pd.DataFrame,
                           combates2_cleaned: pd.DataFrame,
                           combates3_cleaned: pd.DataFrame,
                           card_master_list: pd.DataFrame) -> Dict[str, Any]:
    """Analizar las cartas más utilizadas en los mazos.
    
    Args:
        combates1_cleaned: Dataset de combates 1 limpio
        combates2_cleaned: Dataset de combates 2 limpio
        combates3_cleaned: Dataset de combates 3 limpio
        card_master_list: Lista maestra de cartas con IDs y nombres
        
    Returns:
        Diccionario con análisis de cartas más utilizadas
    """
    logger.info("Analizando cartas más utilizadas...")
    
    # Combinar todos los datasets limpios
    all_combates = pd.concat([combates1_cleaned, combates2_cleaned, combates3_cleaned], 
                            ignore_index=True)
    
    # Identificar columnas de cartas (patrón: winner.card1.id, winner.card2.id, etc.)
    card_columns = [col for col in all_combates.columns if 'card' in col.lower() and 'id' in col.lower()]
    
    if not card_columns:
        logger.warning("No se encontraron columnas de cartas con el patrón esperado")
        return {'error': 'No se encontraron columnas de cartas'}
    
    # Contar frecuencia de cada carta
    all_cards = []
    
    for col in card_columns:
        cards_in_column = all_combates[col].dropna().tolist()
        all_cards.extend(cards_in_column)
    
    # Análisis de frecuencia
    card_frequency = Counter(all_cards)
    total_card_uses = len(all_cards)
    unique_cards = len(card_frequency)
    
    # Top 20 cartas más utilizadas
    top_cards = dict(card_frequency.most_common(20))
    
    # Crear diccionario de mapeo ID -> Nombre
    card_id_to_name = {}
    if 'team.card1.id' in card_master_list.columns and 'team.card1.name' in card_master_list.columns:
        card_id_to_name = dict(zip(card_master_list['team.card1.id'], card_master_list['team.card1.name']))
    
    # Calcular porcentajes y agregar nombres de cartas
    top_cards_with_percentage = {}
    for card_id, count in top_cards.items():
        percentage = (count / total_card_uses) * 100
        card_name = card_id_to_name.get(card_id, f"Carta_{card_id}")
        
        top_cards_with_percentage[card_id] = {
            'card_name': card_name,
            'count': count,
            'percentage': round(percentage, 2)
        }
    
    result = {
        'card_usage_stats': {
            'total_card_uses': total_card_uses,
            'unique_cards': unique_cards,
            'average_uses_per_card': round(total_card_uses / unique_cards, 2)
        },
        'top_cards': top_cards_with_percentage,
        'card_columns_found': card_columns,
        'card_mapping_used': len(card_id_to_name) > 0
    }
    
    logger.info(f"Análisis de cartas completado: {unique_cards} cartas únicas, {total_card_uses} usos totales")
    return result


def analyze_win_conditions_usage(combates1_cleaned: pd.DataFrame,
                                combates2_cleaned: pd.DataFrame,
                                combates3_cleaned: pd.DataFrame,
                                wincons: pd.DataFrame,
                                card_master_list: pd.DataFrame) -> Dict[str, Any]:
    """Analizar las win conditions más utilizadas.
    
    Args:
        combates1_cleaned: Dataset de combates 1 limpio
        combates2_cleaned: Dataset de combates 2 limpio
        combates3_cleaned: Dataset de combates 3 limpio
        wincons: Lista de win conditions con IDs y nombres
        card_master_list: Lista maestra de cartas con IDs y nombres
        
    Returns:
        Diccionario con análisis de win conditions más utilizadas
    """
    logger.info("Analizando win conditions más utilizadas...")
    
    # Combinar todos los datasets limpios
    all_combates = pd.concat([combates1_cleaned, combates2_cleaned, combates3_cleaned], 
                            ignore_index=True)
    
    # Obtener lista de IDs de win conditions
    win_condition_ids = set()
    if 'card_id' in wincons.columns:
        win_condition_ids = set(wincons['card_id'].tolist())
    
    # Crear diccionario de mapeo ID -> Nombre para win conditions
    wc_id_to_name = {}
    if 'card_id' in wincons.columns and 'card_name' in wincons.columns:
        wc_id_to_name = dict(zip(wincons['card_id'], wincons['card_name']))
    
    # Crear diccionario de mapeo ID -> Nombre para todas las cartas
    card_id_to_name = {}
    if 'team.card1.id' in card_master_list.columns and 'team.card1.name' in card_master_list.columns:
        card_id_to_name = dict(zip(card_master_list['team.card1.id'], card_master_list['team.card1.name']))
    
    # Identificar columnas de cartas
    card_columns = [col for col in all_combates.columns if 'card' in col.lower() and 'id' in col.lower()]
    
    if not card_columns:
        logger.warning("No se encontraron columnas de cartas")
        return {'error': 'No se encontraron columnas de cartas'}
    
    # Filtrar solo win conditions en las cartas de ganadores
    winner_win_conditions = []
    for col in card_columns:
        if 'winner' in col:
            cards_in_column = all_combates[col].dropna().tolist()
            # Filtrar solo las que son win conditions
            wc_cards = [card for card in cards_in_column if card in win_condition_ids]
            winner_win_conditions.extend(wc_cards)
    
    # Filtrar solo win conditions en las cartas de perdedores
    loser_win_conditions = []
    for col in card_columns:
        if 'loser' in col:
            cards_in_column = all_combates[col].dropna().tolist()
            # Filtrar solo las que son win conditions
            wc_cards = [card for card in cards_in_column if card in win_condition_ids]
            loser_win_conditions.extend(wc_cards)
    
    # Análisis de win conditions más utilizadas por ganadores
    winner_wc_frequency = Counter(winner_win_conditions)
    top_winner_wcs = dict(winner_wc_frequency.most_common(20))
    
    # Análisis de win conditions más utilizadas por perdedores
    loser_wc_frequency = Counter(loser_win_conditions)
    top_loser_wcs = dict(loser_wc_frequency.most_common(20))
    
    # Análisis combinado (todas las win conditions)
    all_wcs = winner_win_conditions + loser_win_conditions
    all_wc_frequency = Counter(all_wcs)
    top_all_wcs = dict(all_wc_frequency.most_common(20))
    
    # Calcular porcentajes y agregar nombres
    total_winner_wc_uses = len(winner_win_conditions)
    total_loser_wc_uses = len(loser_win_conditions)
    total_all_wc_uses = len(all_wcs)
    
    top_winner_wcs_with_percentage = {}
    for wc_id, count in top_winner_wcs.items():
        percentage = (count / total_winner_wc_uses) * 100 if total_winner_wc_uses > 0 else 0
        wc_name = wc_id_to_name.get(wc_id, card_id_to_name.get(wc_id, f"WinCondition_{wc_id}"))
        
        top_winner_wcs_with_percentage[wc_id] = {
            'card_name': wc_name,
            'count': count,
            'percentage': round(percentage, 2)
        }
    
    top_loser_wcs_with_percentage = {}
    for wc_id, count in top_loser_wcs.items():
        percentage = (count / total_loser_wc_uses) * 100 if total_loser_wc_uses > 0 else 0
        wc_name = wc_id_to_name.get(wc_id, card_id_to_name.get(wc_id, f"WinCondition_{wc_id}"))
        
        top_loser_wcs_with_percentage[wc_id] = {
            'card_name': wc_name,
            'count': count,
            'percentage': round(percentage, 2)
        }
    
    top_all_wcs_with_percentage = {}
    for wc_id, count in top_all_wcs.items():
        percentage = (count / total_all_wc_uses) * 100 if total_all_wc_uses > 0 else 0
        wc_name = wc_id_to_name.get(wc_id, card_id_to_name.get(wc_id, f"WinCondition_{wc_id}"))
        
        top_all_wcs_with_percentage[wc_id] = {
            'card_name': wc_name,
            'count': count,
            'percentage': round(percentage, 2)
        }
    
    result = {
        'winner_win_conditions': top_winner_wcs_with_percentage,
        'loser_win_conditions': top_loser_wcs_with_percentage,
        'overall_win_conditions': top_all_wcs_with_percentage,
        'usage_stats': {
            'total_winner_wc_uses': total_winner_wc_uses,
            'total_loser_wc_uses': total_loser_wc_uses,
            'total_wc_uses': total_all_wc_uses,
            'unique_win_conditions': len(all_wc_frequency),
            'total_win_conditions_available': len(win_condition_ids)
        }
    }
    
    logger.info(f"Análisis de win conditions completado: {len(all_wc_frequency)} win conditions únicas")
    return result


def generate_eda_summary(rarity_distributions: Dict[str, Any],
                        most_used_cards: Dict[str, Any],
                        win_conditions_usage: Dict[str, Any]) -> Dict[str, Any]:
    """Generar resumen ejecutivo del EDA.
    
    Args:
        rarity_distributions: Análisis de distribución de rarezas
        most_used_cards: Análisis de cartas más utilizadas
        win_conditions_usage: Análisis de win conditions más utilizadas
        
    Returns:
        Diccionario con resumen ejecutivo del EDA
    """
    logger.info("Generando resumen ejecutivo del EDA...")
    
    summary = {
        'eda_overview': {
            'dataset_size': rarity_distributions['total_records_analyzed'],
            'analysis_focus': ['Cartas más usadas', 'Win conditions más usadas', 'Distribución de rarezas']
        },
        'key_findings': {
            'most_popular_cards': [most_used_cards['top_cards'][card_id]['card_name'] for card_id in list(most_used_cards['top_cards'].keys())[:5]] if 'top_cards' in most_used_cards else [],
            'total_unique_cards': most_used_cards['card_usage_stats']['unique_cards'] if 'card_usage_stats' in most_used_cards else 0,
            'total_win_conditions': win_conditions_usage['usage_stats']['unique_win_conditions'] if 'usage_stats' in win_conditions_usage else 0
        },
        'rarity_insights': {},
        'insights': [
            f"Dataset contiene {rarity_distributions['total_records_analyzed']:,} registros de batallas",
            f"Se identificaron {most_used_cards['card_usage_stats']['unique_cards'] if 'card_usage_stats' in most_used_cards else 0} cartas únicas",
            f"Total de win conditions analizadas: {win_conditions_usage['usage_stats']['unique_win_conditions'] if 'usage_stats' in win_conditions_usage else 0}"
        ]
    }
    
    # Agregar insights específicos si hay datos disponibles
    if 'top_cards' in most_used_cards and most_used_cards['top_cards']:
        top_card_id = list(most_used_cards['top_cards'].keys())[0]
        top_card_name = most_used_cards['top_cards'][top_card_id]['card_name']
        top_percentage = most_used_cards['top_cards'][top_card_id]['percentage']
        summary['insights'].append(f"Carta más popular: {top_card_name} ({top_percentage}% de uso)")
    
    if 'overall_win_conditions' in win_conditions_usage and win_conditions_usage['overall_win_conditions']:
        top_wc_id = list(win_conditions_usage['overall_win_conditions'].keys())[0]
        top_wc_name = win_conditions_usage['overall_win_conditions'][top_wc_id]['card_name']
        top_wc_percentage = win_conditions_usage['overall_win_conditions'][top_wc_id]['percentage']
        summary['insights'].append(f"Win condition más usada: {top_wc_name} ({top_wc_percentage}% de uso)")
    
    # Agregar insights de rarezas
    if 'winner_vs_loser_comparison' in rarity_distributions:
        for rarity, stats in rarity_distributions['winner_vs_loser_comparison'].items():
            difference = stats['difference']
            if difference > 0:
                summary['insights'].append(f"Ganadores usan más cartas {rarity}: +{difference:.2f} promedio")
            elif difference < 0:
                summary['insights'].append(f"Perdedores usan más cartas {rarity}: +{abs(difference):.2f} promedio")
    
    logger.info("Resumen ejecutivo del EDA generado exitosamente")
    return summary

