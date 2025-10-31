"""Nodos para el pipeline de regresión."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import logging

logger = logging.getLogger(__name__)


def _extract_features_and_labels(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extraer features y labels del dataset.
    
    Args:
        data: DataFrame con features y labels
        
    Returns:
        Tuple con (X, y)
    """
    X = data.drop(columns=['label', 'target_regression']).copy()
    y = data['target_regression'].copy()
    return X, y


def save_model_metrics(metrics: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Guardar métricas del modelo en formato serializable.
    
    Args:
        metrics: Diccionario con métricas
        model_name: Nombre del modelo
        
    Returns:
        Diccionario con métricas convertidas a tipos básicos
    """
    # Convertir numpy types a Python types para JSON
    return {
        'model_name': model_name,
        'mae': float(metrics['mae']),
        'mse': float(metrics['mse']),
        'rmse': float(metrics['rmse']),
        'r2': float(metrics['r2']),
        'best_params': metrics['best_params']
    }


def train_linear_regression(train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de Linear Regression con GridSearchCV.
    
    Args:
        train_data: Data de entrenamiento
        test_data: Data de test
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelo y métricas
    """
    logger.info("Entrenando Linear Regression...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    model = LinearRegression()
    param_grid = parameters['models']['linear_regression']['param_grid']
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"Linear Regression - Best params: {grid_search.best_params_}, R²: {metrics['r2']:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_model_metrics(metrics, 'Linear Regression')
    
    return {
        'model': best_model,
        'metrics': metrics
    }, metrics_json


def train_ridge(train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de Ridge con GridSearchCV.
    
    Args:
        train_data: Data de entrenamiento
        test_data: Data de test
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelo y métricas
    """
    logger.info("Entrenando Ridge...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    model = Ridge(random_state=parameters['random_state'])
    param_grid = parameters['models']['ridge']['param_grid']
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"Ridge - Best params: {grid_search.best_params_}, R²: {metrics['r2']:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_model_metrics(metrics, 'Ridge')
    
    return {
        'model': best_model,
        'metrics': metrics
    }, metrics_json


def train_random_forest(train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de Random Forest con GridSearchCV.
    
    Args:
        train_data: Data de entrenamiento
        test_data: Data de test
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelo y métricas
    """
    logger.info("Entrenando Random Forest Regressor...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    model = RandomForestRegressor(random_state=parameters['random_state'])
    param_grid = parameters['models']['random_forest']['param_grid']
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"Random Forest Regressor - Best params: {grid_search.best_params_}, R²: {metrics['r2']:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_model_metrics(metrics, 'Random Forest')
    
    return {
        'model': best_model,
        'metrics': metrics
    }, metrics_json


def train_xgboost(train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de XGBoost Regressor con GridSearchCV.
    
    Args:
        train_data: Data de entrenamiento
        test_data: Data de test
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelo y métricas
    """
    logger.info("Entrenando XGBoost Regressor...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    model = xgb.XGBRegressor(random_state=parameters['random_state'])
    param_grid = parameters['models']['xgboost']['param_grid']
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"XGBoost Regressor - Best params: {grid_search.best_params_}, R²: {metrics['r2']:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_model_metrics(metrics, 'XGBoost')
    
    return {
        'model': best_model,
        'metrics': metrics
    }, metrics_json


def train_svr(train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de SVR (optimizado con LinearSVR) con GridSearchCV.
    
    Args:
        train_data: Data de entrenamiento
        test_data: Data de test
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelo y métricas
    """
    logger.info("Entrenando SVR...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    # LinearSVR es considerablemente más rápido que SVR con kernel RBF
    model = LinearSVR(random_state=parameters['random_state'], max_iter=2000)
    # Usamos solo C y epsilon del parámetro, ignorando el kernel
    raw_grid = parameters['models']['svr']['param_grid']
    param_grid = {
        'C': raw_grid.get('C', [1.0]),
        'epsilon': raw_grid.get('epsilon', [0.1])
    }
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"SVR - Best params: {grid_search.best_params_}, R²: {metrics['r2']:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_model_metrics(metrics, 'SVR')
    
    return {
        'model': best_model,
        'metrics': metrics
    }, metrics_json


def train_lightgbm(train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de LightGBM Regressor con GridSearchCV.
    
    Args:
        train_data: Data de entrenamiento
        test_data: Data de test
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelo y métricas
    """
    logger.info("Entrenando LightGBM Regressor...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    model = lgb.LGBMRegressor(random_state=parameters['random_state'], verbose=-1)
    param_grid = parameters['models']['lightgbm']['param_grid']
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"LightGBM Regressor - Best params: {grid_search.best_params_}, R²: {metrics['r2']:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_model_metrics(metrics, 'LightGBM')
    
    return {
        'model': best_model,
        'metrics': metrics
    }, metrics_json


def create_regression_comparison(
    linear_result: Dict[str, Any],
    ridge_result: Dict[str, Any],
    rf_reg_result: Dict[str, Any],
    xgb_reg_result: Dict[str, Any],
    svr_result: Dict[str, Any],
    lgbm_reg_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Crear tabla comparativa de todos los modelos de regresión.
    
    Args:
        linear_result: Resultados de Linear Regression
        ridge_result: Resultados de Ridge
        rf_reg_result: Resultados de Random Forest
        xgb_reg_result: Resultados de XGBoost
        svr_result: Resultados de SVR
        lgbm_reg_result: Resultados de LightGBM
        
        Returns:
        Diccionario con comparación
    """
    logger.info("Creando comparación de modelos de regresión...")
    
    comparison = {
        'models': {
            'Linear Regression': {
                'metrics': linear_result['metrics'],
                'best_params': linear_result['metrics']['best_params']
            },
            'Ridge': {
                'metrics': ridge_result['metrics'],
                'best_params': ridge_result['metrics']['best_params']
            },
            'Random Forest': {
                'metrics': rf_reg_result['metrics'],
                'best_params': rf_reg_result['metrics']['best_params']
            },
            'XGBoost': {
                'metrics': xgb_reg_result['metrics'],
                'best_params': xgb_reg_result['metrics']['best_params']
            },
            'SVR': {
                'metrics': svr_result['metrics'],
                'best_params': svr_result['metrics']['best_params']
            },
            'LightGBM': {
                'metrics': lgbm_reg_result['metrics'],
                'best_params': lgbm_reg_result['metrics']['best_params']
            }
        }
    }
    
    logger.info("Comparación de regresión creada exitosamente")
    return comparison


def consolidate_regression_metrics(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidar métricas de regresión en formato JSON.
    
    Args:
        comparison: Comparación de modelos
        
    Returns:
        Diccionario con métricas consolidadas
    """
    logger.info("Consolidando métricas de regresión...")
    
    metrics_dict = {}
    
    for model_name, model_data in comparison['models'].items():
        metrics_dict[model_name] = {
            'mae': float(model_data['metrics']['mae']),
            'mse': float(model_data['metrics']['mse']),
            'rmse': float(model_data['metrics']['rmse']),
            'r2': float(model_data['metrics']['r2']),
            'best_params': model_data['best_params']
        }
    
    # Determinar mejor modelo (menor MAE)
    best_model = min(metrics_dict.items(), key=lambda x: x[1]['mae'])
    metrics_dict['_best_model'] = {'name': best_model[0], 'mae': best_model[1]['mae']}
    
    logger.info(f"Mejor modelo de regresión: {best_model[0]} (MAE: {best_model[1]['mae']:.4f})")
    
    return metrics_dict
