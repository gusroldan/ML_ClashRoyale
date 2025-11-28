"""Nodos para análisis SHAP (SHapley Additive exPlanations) de modelos ML."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
import json
import pickle
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP no está disponible. Instala con: pip install shap>=0.42.0")

logger = logging.getLogger(__name__)


def calculate_shap_values_classification(
    model_result: Dict[str, Any],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Calcular SHAP values para modelos de clasificación.
    
    Args:
        model_result: Resultado del modelo entrenado (dict con 'model')
        train_data: Datos de entrenamiento
        test_data: Datos de test
        model_name: Nombre del modelo
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con SHAP values y análisis
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP no está disponible. Saltando análisis SHAP.")
        return {'shap_available': False}
    
    # Extraer nombre del modelo desde el resultado o parámetros
    model_name = parameters.get('model_name', 'Unknown Model')
    
    logger.info(f"Calculando SHAP values para {model_name}...")
    
    # Extraer modelo y datos
    model = model_result['model']
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    # Configuración SHAP
    n_samples_explain = parameters.get('shap', {}).get('n_samples_explain', min(100, len(X_test)))
    n_background_samples = parameters.get('shap', {}).get('n_background_samples', min(100, len(X_train)))
    
    # Seleccionar muestras para explicación (para eficiencia)
    if len(X_test) > n_samples_explain:
        sample_indices = np.random.choice(len(X_test), n_samples_explain, replace=False)
        X_explain = X_test.iloc[sample_indices].copy()
    else:
        X_explain = X_test.copy()
        sample_indices = np.arange(len(X_test))
    
    # Seleccionar muestras de fondo (background)
    if len(X_train) > n_background_samples:
        background_indices = np.random.choice(len(X_train), n_background_samples, replace=False)
        X_background = X_train.iloc[background_indices].copy()
    else:
        X_background = X_train.copy()
    
    try:
        # Determinar el tipo de explainer según el modelo
        model_type = _get_model_type(model)
        
        if model_type == 'tree':
            # Para modelos basados en árboles (Random Forest, XGBoost, LightGBM)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
            
            # Si es clasificación binaria, shap_values puede ser una lista
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Usar valores para clase positiva
            
        elif model_type == 'linear':
            # Para modelos lineales (Logistic Regression, LinearSVC)
            explainer = shap.LinearExplainer(model, X_background)
            shap_values = explainer.shap_values(X_explain)
            
        else:
            # Para otros modelos, usar KernelExplainer (más lento pero universal)
            logger.info(f"Usando KernelExplainer para {model_name} (puede ser lento)...")
            explainer = shap.KernelExplainer(model.predict_proba, X_background)
            shap_values = explainer.shap_values(X_explain)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        # Calcular importancia global de features
        feature_importance = pd.DataFrame({
            'feature': X_explain.columns,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0),
            'std_shap': shap_values.std(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Top features más importantes
        top_n = parameters.get('shap', {}).get('top_n_features', 20)
        top_features = feature_importance.head(top_n).to_dict('records')
        
        # Estadísticas de SHAP values
        shap_stats = {
            'mean_shap_value': float(np.mean(shap_values)),
            'std_shap_value': float(np.std(shap_values)),
            'min_shap_value': float(np.min(shap_values)),
            'max_shap_value': float(np.max(shap_values)),
            'n_samples_explained': int(len(X_explain)),
            'n_features': int(len(X_explain.columns))
        }
        
        logger.info(f"SHAP values calculados para {model_name}: {len(X_explain)} muestras, {len(X_explain.columns)} features")
        
        return {
            'shap_available': True,
            'model_name': model_name,
            'explainer_type': model_type,
            'feature_importance': top_features,
            'shap_statistics': shap_stats,
            'n_samples_explained': int(len(X_explain)),
            'top_n_features': top_n
        }
        
    except Exception as e:
        logger.error(f"Error calculando SHAP values para {model_name}: {str(e)}")
        return {
            'shap_available': False,
            'error': str(e),
            'model_name': model_name
        }


def calculate_shap_values_regression(
    model_result: Dict[str, Any],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Calcular SHAP values para modelos de regresión.
    
    Args:
        model_result: Resultado del modelo entrenado (dict con 'model')
        train_data: Datos de entrenamiento
        test_data: Datos de test
        model_name: Nombre del modelo
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con SHAP values y análisis
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP no está disponible. Saltando análisis SHAP.")
        return {'shap_available': False}
    
    # Extraer nombre del modelo desde el resultado o parámetros
    model_name = parameters.get('model_name', 'Unknown Model')
    
    logger.info(f"Calculando SHAP values para {model_name} (regresión)...")
    
    # Extraer modelo y datos
    model = model_result['model']
    X_train, y_train = _extract_features_and_labels_regression(train_data)
    X_test, y_test = _extract_features_and_labels_regression(test_data)
    
    # Configuración SHAP
    n_samples_explain = parameters.get('shap', {}).get('n_samples_explain', min(100, len(X_test)))
    n_background_samples = parameters.get('shap', {}).get('n_background_samples', min(100, len(X_train)))
    
    # Seleccionar muestras para explicación
    if len(X_test) > n_samples_explain:
        sample_indices = np.random.choice(len(X_test), n_samples_explain, replace=False)
        X_explain = X_test.iloc[sample_indices].copy()
    else:
        X_explain = X_test.copy()
        sample_indices = np.arange(len(X_test))
    
    # Seleccionar muestras de fondo
    if len(X_train) > n_background_samples:
        background_indices = np.random.choice(len(X_train), n_background_samples, replace=False)
        X_background = X_train.iloc[background_indices].copy()
    else:
        X_background = X_train.copy()
    
    try:
        # Determinar el tipo de explainer según el modelo
        model_type = _get_model_type(model)
        
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
            
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_background)
            shap_values = explainer.shap_values(X_explain)
            
        else:
            logger.info(f"Usando KernelExplainer para {model_name} (puede ser lento)...")
            explainer = shap.KernelExplainer(model.predict, X_background)
            shap_values = explainer.shap_values(X_explain)
        
        # Calcular importancia global de features
        feature_importance = pd.DataFrame({
            'feature': X_explain.columns,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0),
            'std_shap': shap_values.std(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Top features más importantes
        top_n = parameters.get('shap', {}).get('top_n_features', 20)
        top_features = feature_importance.head(top_n).to_dict('records')
        
        # Estadísticas de SHAP values
        shap_stats = {
            'mean_shap_value': float(np.mean(shap_values)),
            'std_shap_value': float(np.std(shap_values)),
            'min_shap_value': float(np.min(shap_values)),
            'max_shap_value': float(np.max(shap_values)),
            'n_samples_explained': int(len(X_explain)),
            'n_features': int(len(X_explain.columns))
        }
        
        logger.info(f"SHAP values calculados para {model_name}: {len(X_explain)} muestras, {len(X_explain.columns)} features")
        
        return {
            'shap_available': True,
            'model_name': model_name,
            'explainer_type': model_type,
            'feature_importance': top_features,
            'shap_statistics': shap_stats,
            'n_samples_explained': int(len(X_explain)),
            'top_n_features': top_n
        }
        
    except Exception as e:
        logger.error(f"Error calculando SHAP values para {model_name}: {str(e)}")
        return {
            'shap_available': False,
            'error': str(e),
            'model_name': model_name
        }


def create_shap_summary_classification(
    rf_shap: Dict[str, Any],
    xgb_shap: Dict[str, Any],
    lgbm_shap: Dict[str, Any]
) -> Dict[str, Any]:
    """Crear resumen de análisis SHAP para todos los modelos de clasificación.
    
    Args:
        rf_shap: Resultados SHAP de Random Forest
        xgb_shap: Resultados SHAP de XGBoost
        lgbm_shap: Resultados SHAP de LightGBM
        
    Returns:
        Diccionario con resumen comparativo
    """
    logger.info("Creando resumen de análisis SHAP para clasificación...")
    
    # Crear diccionario con resultados
    shap_results = {
        "Random Forest": rf_shap,
        "XGBoost": xgb_shap,
        "LightGBM": lgbm_shap
    }
    
    summary = {
        'models_analyzed': [],
        'top_features_across_models': {},
        'model_comparison': []
    }
    
    # Recopilar top features de todos los modelos
    all_features = {}
    for model_name, result in shap_results.items():
        if result.get('shap_available', False) and 'feature_importance' in result:
            summary['models_analyzed'].append(model_name)
            
            # Agregar features importantes
            for feat_info in result['feature_importance']:
                feat_name = feat_info['feature']
                if feat_name not in all_features:
                    all_features[feat_name] = []
                all_features[feat_name].append({
                    'model': model_name,
                    'mean_abs_shap': feat_info['mean_abs_shap']
                })
            
            # Agregar comparación de modelos
            summary['model_comparison'].append({
                'model_name': model_name,
                'explainer_type': result.get('explainer_type', 'unknown'),
                'n_samples_explained': result.get('n_samples_explained', 0),
                'top_feature': result['feature_importance'][0]['feature'] if result['feature_importance'] else None,
                'top_feature_importance': result['feature_importance'][0]['mean_abs_shap'] if result['feature_importance'] else 0.0
            })
    
    # Calcular importancia promedio de features across models
    for feat_name, model_scores in all_features.items():
        avg_importance = np.mean([m['mean_abs_shap'] for m in model_scores])
        summary['top_features_across_models'][feat_name] = {
            'average_importance': float(avg_importance),
            'models_count': len(model_scores),
            'models': [m['model'] for m in model_scores]
        }
    
    # Ordenar features por importancia promedio
    summary['top_features_across_models'] = dict(
        sorted(
            summary['top_features_across_models'].items(),
            key=lambda x: x[1]['average_importance'],
            reverse=True
        )[:20]  # Top 20
    )
    
    logger.info(f"Resumen SHAP creado para {len(summary['models_analyzed'])} modelos")
    
    return summary


def create_shap_summary_regression(
    rf_shap: Dict[str, Any],
    xgb_shap: Dict[str, Any],
    lgbm_shap: Dict[str, Any]
) -> Dict[str, Any]:
    """Crear resumen de análisis SHAP para todos los modelos de regresión.
    
    Args:
        rf_shap: Resultados SHAP de Random Forest
        xgb_shap: Resultados SHAP de XGBoost
        lgbm_shap: Resultados SHAP de LightGBM
        
    Returns:
        Diccionario con resumen comparativo
    """
    logger.info("Creando resumen de análisis SHAP para regresión...")
    
    # Misma lógica que clasificación
    return create_shap_summary_classification(rf_shap, xgb_shap, lgbm_shap)


def _extract_features_and_labels(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extraer features y labels del dataset (clasificación).
    
    Args:
        data: DataFrame con features y labels
        
    Returns:
        Tuple con (X, y)
    """
    X = data.drop(columns=['label', 'target_regression']).copy()
    y = data['label'].copy()
    return X, y


def _extract_features_and_labels_regression(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extraer features y labels del dataset (regresión).
    
    Args:
        data: DataFrame con features y labels
        
    Returns:
        Tuple con (X, y)
    """
    X = data.drop(columns=['label', 'target_regression']).copy()
    y = data['target_regression'].copy()
    return X, y


def _get_model_type(model) -> str:
    """Determinar el tipo de modelo para seleccionar el explainer SHAP apropiado.
    
    Args:
        model: Modelo entrenado
        
    Returns:
        Tipo de modelo: 'tree', 'linear', o 'other'
    """
    model_class = type(model).__name__
    
    # Modelos basados en árboles
    tree_models = [
        'RandomForestClassifier', 'RandomForestRegressor',
        'XGBClassifier', 'XGBRegressor',
        'LGBMClassifier', 'LGBMRegressor',
        'GradientBoostingClassifier', 'GradientBoostingRegressor',
        'DecisionTreeClassifier', 'DecisionTreeRegressor'
    ]
    
    # Modelos lineales
    linear_models = [
        'LogisticRegression', 'LinearRegression',
        'Ridge', 'Lasso', 'ElasticNet',
        'LinearSVC', 'LinearSVR',
        'SGDClassifier', 'SGDRegressor'
    ]
    
    if any(tree_model in model_class for tree_model in tree_models):
        return 'tree'
    elif any(linear_model in model_class for linear_model in linear_models):
        return 'linear'
    else:
        return 'other'

