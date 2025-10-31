"""Nodos para feature engineering del pipeline ML."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def create_card_dictionary(card_master_list: pd.DataFrame) -> Dict[int, str]:
    """Crear diccionario de mapeo de IDs de cartas a nombres.
    
    Args:
        card_master_list: DataFrame con lista maestra de cartas
        
    Returns:
        Diccionario {card_id: card_name}
    """
    logger.info("Creando diccionario de cartas...")
    
    # Extraer columnas de IDs y nombres (asumiendo que tienen patrón team.card1.id/name)
    card_ids = []
    card_names = []
    
    for col in card_master_list.columns:
        if 'id' in col and col != 'team.card1.id':
            base = col.replace('.id', '').replace('team.', '')
            name_col = base + '.name'
            if name_col in card_master_list.columns:
                ids = card_master_list[col].dropna().unique()
                names = card_master_list[name_col].dropna().unique()
                card_ids.extend(ids)
                card_names.extend(names)
    
    card_dict = dict(zip(card_ids, card_names))
    
    # Asegurar que todas las cartas del 0 al 101 están en el diccionario
    # (hay 102 cartas totales)
    logger.info(f"Dictionary creado con {len(card_dict)} cartas")
    
    return card_dict


def create_balanced_dataset(combined_dataset: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Crear dataset balanceado con re-etiquetado 50/50.
    
    Re-etiqueta: 50% filas donde A=winner, 50% donde A=loser.
    
    Args:
        combined_dataset: Dataset combinado original
        
    Returns:
        Dataset balanceado con columnas re-etiquetadas para A y B
    """
    logger.info("Creando dataset balanceado con re-etiquetado 50/50...")
    logger.info(f"Dataset original tiene {len(combined_dataset)} registros, muestrearemos a 1M para reducir memoria")
    
    # Crear dos datasets: uno donde A gana y otro donde A pierde
    # Dataset 1: A gana (A = winner)
    df_a_wins = combined_dataset.copy()
    df_a_wins['A_tag'] = df_a_wins['winner.tag']
    df_a_wins['A_startingTrophies'] = df_a_wins['winner.startingTrophies']
    df_a_wins['A_trophyChange'] = df_a_wins['winner.trophyChange']
    df_a_wins['A_card1.id'] = df_a_wins['winner.card1.id']
    df_a_wins['A_card2.id'] = df_a_wins['winner.card2.id']
    df_a_wins['A_card3.id'] = df_a_wins['winner.card3.id']
    df_a_wins['A_card4.id'] = df_a_wins['winner.card4.id']
    df_a_wins['A_card5.id'] = df_a_wins['winner.card5.id']
    df_a_wins['A_card6.id'] = df_a_wins['winner.card6.id']
    df_a_wins['A_card7.id'] = df_a_wins['winner.card7.id']
    df_a_wins['A_card8.id'] = df_a_wins['winner.card8.id']
    df_a_wins['A_common.count'] = df_a_wins['winner.common.count']
    df_a_wins['A_rare.count'] = df_a_wins['winner.rare.count']
    df_a_wins['A_epic.count'] = df_a_wins['winner.epic.count']
    df_a_wins['A_legendary.count'] = df_a_wins['winner.legendary.count']
    
    df_a_wins['B_tag'] = df_a_wins['loser.tag']
    df_a_wins['B_startingTrophies'] = df_a_wins['loser.startingTrophies']
    df_a_wins['B_trophyChange'] = df_a_wins['loser.trophyChange']
    df_a_wins['B_card1.id'] = df_a_wins['loser.card1.id']
    df_a_wins['B_card2.id'] = df_a_wins['loser.card2.id']
    df_a_wins['B_card3.id'] = df_a_wins['loser.card3.id']
    df_a_wins['B_card4.id'] = df_a_wins['loser.card4.id']
    df_a_wins['B_card5.id'] = df_a_wins['loser.card5.id']
    df_a_wins['B_card6.id'] = df_a_wins['loser.card6.id']
    df_a_wins['B_card7.id'] = df_a_wins['loser.card7.id']
    df_a_wins['B_card8.id'] = df_a_wins['loser.card8.id']
    df_a_wins['B_common.count'] = df_a_wins['loser.common.count']
    df_a_wins['B_rare.count'] = df_a_wins['loser.rare.count']
    df_a_wins['B_epic.count'] = df_a_wins['loser.epic.count']
    df_a_wins['B_legendary.count'] = df_a_wins['loser.legendary.count']
    
    df_a_wins['label'] = 1  # A gana
    df_a_wins['target_regression'] = df_a_wins['A_trophyChange']
    
    # Dataset 2: A pierde (A = loser)
    df_a_loses = combined_dataset.copy()
    df_a_loses['A_tag'] = df_a_loses['loser.tag']
    df_a_loses['A_startingTrophies'] = df_a_loses['loser.startingTrophies']
    df_a_loses['A_trophyChange'] = df_a_loses['loser.trophyChange']
    df_a_loses['A_card1.id'] = df_a_loses['loser.card1.id']
    df_a_loses['A_card2.id'] = df_a_loses['loser.card2.id']
    df_a_loses['A_card3.id'] = df_a_loses['loser.card3.id']
    df_a_loses['A_card4.id'] = df_a_loses['loser.card4.id']
    df_a_loses['A_card5.id'] = df_a_loses['loser.card5.id']
    df_a_loses['A_card6.id'] = df_a_loses['loser.card6.id']
    df_a_loses['A_card7.id'] = df_a_loses['loser.card7.id']
    df_a_loses['A_card8.id'] = df_a_loses['loser.card8.id']
    df_a_loses['A_common.count'] = df_a_loses['loser.common.count']
    df_a_loses['A_rare.count'] = df_a_loses['loser.rare.count']
    df_a_loses['A_epic.count'] = df_a_loses['loser.epic.count']
    df_a_loses['A_legendary.count'] = df_a_loses['loser.legendary.count']
    
    df_a_loses['B_tag'] = df_a_loses['winner.tag']
    df_a_loses['B_startingTrophies'] = df_a_loses['winner.startingTrophies']
    df_a_loses['B_trophyChange'] = df_a_loses['winner.trophyChange']
    df_a_loses['B_card1.id'] = df_a_loses['winner.card1.id']
    df_a_loses['B_card2.id'] = df_a_loses['winner.card2.id']
    df_a_loses['B_card3.id'] = df_a_loses['winner.card3.id']
    df_a_loses['B_card4.id'] = df_a_loses['winner.card4.id']
    df_a_loses['B_card5.id'] = df_a_loses['winner.card5.id']
    df_a_loses['B_card6.id'] = df_a_loses['winner.card6.id']
    df_a_loses['B_card7.id'] = df_a_loses['winner.card7.id']
    df_a_loses['B_card8.id'] = df_a_loses['winner.card8.id']
    df_a_loses['B_common.count'] = df_a_loses['winner.common.count']
    df_a_loses['B_rare.count'] = df_a_loses['winner.rare.count']
    df_a_loses['B_epic.count'] = df_a_loses['winner.epic.count']
    df_a_loses['B_legendary.count'] = df_a_loses['winner.legendary.count']
    
    df_a_loses['label'] = 0  # A pierde
    df_a_loses['target_regression'] = df_a_loses['A_trophyChange']
    
    # Balancear: tomar muestras iguales de ambos datasets
    # Limitar muestras por clase para evitar problemas de memoria
    max_samples_per_class = parameters.get('max_samples_per_class', 500000)
    n_samples = min(len(df_a_wins), len(df_a_loses), max_samples_per_class)
    
    df_a_wins_sample = df_a_wins.sample(n=n_samples, random_state=42)
    df_a_loses_sample = df_a_loses.sample(n=n_samples, random_state=42)
    
    # Combinar
    balanced_dataset = pd.concat([df_a_wins_sample, df_a_loses_sample], ignore_index=True)
    
    # Mezclar
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Dataset balanceado creado: {len(balanced_dataset)} registros ({balanced_dataset['label'].sum()} con label=1, {len(balanced_dataset)-balanced_dataset['label'].sum()} con label=0)")
    
    return balanced_dataset


def encode_cards_multihot(balanced_dataset: pd.DataFrame, card_dictionary: Dict[int, str]) -> pd.DataFrame:
    """Crear features multi-hot encoding de cartas.
    
    Args:
        balanced_dataset: Dataset balanceado
        card_dictionary: Diccionario de cartas
        
    Returns:
        DataFrame con features multi-hot y cards_diff
    """
    logger.info("Creando features multi-hot encoding de cartas...")
    
    # Obtener todos los IDs de cartas únicos del dataset
    all_card_ids = set()
    for i in range(1, 9):
        all_card_ids.update(balanced_dataset[f'A_card{i}.id'].dropna().unique())
        all_card_ids.update(balanced_dataset[f'B_card{i}.id'].dropna().unique())
    
    num_cards = len(all_card_ids)
    logger.info(f"Total de cartas únicas: {num_cards}")
    
    # Crear mapping de card_id a índice (0 a num_cards-1)
    card_to_idx = {card_id: idx for idx, card_id in enumerate(sorted(all_card_ids))}
    
    # Vectorizar cartas de A y B
    vec_a = np.zeros((len(balanced_dataset), num_cards))
    vec_b = np.zeros((len(balanced_dataset), num_cards))
    
    for i in range(1, 9):
        col_a = f'A_card{i}.id'
        col_b = f'B_card{i}.id'
        
        if col_a in balanced_dataset.columns and col_b in balanced_dataset.columns:
            for idx, (a_card, b_card) in enumerate(zip(balanced_dataset[col_a], balanced_dataset[col_b])):
                if pd.notna(a_card) and a_card in card_to_idx:
                    vec_a[idx, card_to_idx[a_card]] = 1
                if pd.notna(b_card) and b_card in card_to_idx:
                    vec_b[idx, card_to_idx[b_card]] = 1
    
    # Calcular cards_diff = vec_A - vec_B
    cards_diff = vec_a - vec_b
    
    # Crear DataFrame con features de cartas
    card_features = pd.DataFrame(
        cards_diff,
        columns=[f'card_{i}' for i in range(num_cards)]
    )
    
    # Agregar información de tarjetas
    card_features['battle_id'] = balanced_dataset['battle_id'].values if 'battle_id' in balanced_dataset.columns else range(len(balanced_dataset))
    card_features['label'] = balanced_dataset['label'].values if 'label' in balanced_dataset.columns else None
    card_features['target_regression'] = balanced_dataset['target_regression'].values if 'target_regression' in balanced_dataset.columns else None
    
    logger.info(f"Features multi-hot creados: {num_cards} dimensiones de cartas")
    
    return card_features


def create_aggregate_features(balanced_dataset: pd.DataFrame, features_multihot: pd.DataFrame) -> pd.DataFrame:
    """Crear features agregadas.
    
    Args:
        balanced_dataset: Dataset balanceado original
        features_multihot: Features de cartas multi-hot
        
    Returns:
        DataFrame con features agregadas
    """
    logger.info("Creando features agregadas...")
    
    # Delta de trofeos
    delta_trophies = balanced_dataset['A_startingTrophies'] - balanced_dataset['B_startingTrophies']
    
    # Conteos de rareza
    df_features = pd.DataFrame({
        'delta_trophies': delta_trophies,
        'A_common.count': balanced_dataset['A_common.count'],
        'A_rare.count': balanced_dataset['A_rare.count'],
        'A_epic.count': balanced_dataset['A_epic.count'],
        'A_legendary.count': balanced_dataset['A_legendary.count'],
        'B_common.count': balanced_dataset['B_common.count'],
        'B_rare.count': balanced_dataset['B_rare.count'],
        'B_epic.count': balanced_dataset['B_epic.count'],
        'B_legendary.count': balanced_dataset['B_legendary.count'],
    })
    
    # Agregar otras features del dataset balanceado si existen
    if 'battle_id' in balanced_dataset.columns:
        df_features['battle_id'] = balanced_dataset['battle_id'].values
    if 'label' in balanced_dataset.columns:
        df_features['label'] = balanced_dataset['label'].values
    if 'target_regression' in balanced_dataset.columns:
        df_features['target_regression'] = balanced_dataset['target_regression'].values
    
    logger.info(f"Features agregadas creadas: {len(df_features.columns)} columnas")
    
    return df_features


def combine_features(features_multihot: pd.DataFrame, features_aggregate: pd.DataFrame) -> pd.DataFrame:
    """Combinar todos los features en un solo dataset.
    
    Args:
        features_multihot: Features multi-hot de cartas
        features_aggregate: Features agregadas
        
    Returns:
        Dataset combinado con todos los features
    """
    logger.info("Combinando features...")
    
    # Combinar por battle_id o por índice
    if 'battle_id' in features_multihot.columns and 'battle_id' in features_aggregate.columns:
        combined = pd.merge(
            features_multihot.drop(columns=['label', 'target_regression']),
            features_aggregate,
            on='battle_id',
            how='inner'
        )
    else:
        # Fallback: combinar por índice
        combined = pd.concat([
            features_multihot.drop(columns=['battle_id', 'label', 'target_regression']),
            features_aggregate[['delta_trophies', 'A_common.count', 'A_rare.count', 'A_epic.count', 'A_legendary.count',
                               'B_common.count', 'B_rare.count', 'B_epic.count', 'B_legendary.count']]
        ], axis=1)
        
        if 'battle_id' in features_aggregate.columns:
            combined['battle_id'] = features_aggregate['battle_id'].values
        if 'label' in features_aggregate.columns:
            combined['label'] = features_aggregate['label'].values
        if 'target_regression' in features_aggregate.columns:
            combined['target_regression'] = features_aggregate['target_regression'].values
    
    logger.info(f"Features combinados: {len(combined)} registros, {len(combined.columns)} features")
    
    return combined


def split_train_test(features_combined: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separar dataset en train y test con estratificación.
    
    Args:
        features_combined: Dataset con todos los features
        parameters: Parámetros del split (test_size)
        
    Returns:
        Tupla con (train_data, test_data)
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Separando en train/test...")
    
    test_size = parameters.get('test_size', 0.2)
    
    # Separar features y targets
    X = features_combined.drop(columns=['label', 'target_regression', 'battle_id'] if 'battle_id' in features_combined.columns else ['label', 'target_regression'])
    y_classification = features_combined['label']
    y_regression = features_combined['target_regression']
    
    # Split con estratificación por label de clasificación
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_classification, y_regression,
        test_size=test_size,
        random_state=42,
        stratify=y_classification
    )
    
    # Crear DataFrames completos
    train_data = X_train.copy()
    train_data['label'] = y_train_class.values
    train_data['target_regression'] = y_train_reg.values
    
    test_data = X_test.copy()
    test_data['label'] = y_test_class.values
    test_data['target_regression'] = y_test_reg.values
    
    logger.info(f"Train: {len(train_data)} registros, Test: {len(test_data)} registros")
    logger.info(f"Train - Clasificación: {train_data['label'].sum()} positivos, {len(train_data)-train_data['label'].sum()} negativos")
    
    return train_data, test_data
