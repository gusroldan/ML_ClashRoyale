"""Nodos para el pipeline de clasificación."""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb
import logging
import json

logger = logging.getLogger(__name__)


def _extract_features_and_labels(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extraer features y labels del dataset.
    
    Args:
        data: DataFrame con features y labels
        
    Returns:
        Tuple con (X, y)
    """
    X = data.drop(columns=['label', 'target_regression']).copy()
    y = data['label'].copy()
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
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'roc_auc': float(metrics['roc_auc']),
        'best_params': metrics['best_params']
    }


def train_logistic_regression(train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de Logistic Regression con GridSearchCV.
    
    Args:
        train_data: Data de entrenamiento
        test_data: Data de test
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelo y métricas
    """
    logger.info("Entrenando Logistic Regression...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    model = LogisticRegression(random_state=parameters['random_state'], max_iter=1000)
    param_grid = parameters['models']['logistic_regression']['param_grid']
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"Logistic Regression - Best params: {grid_search.best_params_}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_model_metrics(metrics, 'Logistic Regression')
    
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
    logger.info("Entrenando Random Forest...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    model = RandomForestClassifier(random_state=parameters['random_state'])
    param_grid = parameters['models']['random_forest']['param_grid']
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"Random Forest - Best params: {grid_search.best_params_}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_model_metrics(metrics, 'Random Forest')
    
    return {
        'model': best_model,
        'metrics': metrics
    }, metrics_json


def train_xgboost(train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de XGBoost con GridSearchCV.
    
    Args:
        train_data: Data de entrenamiento
        test_data: Data de test
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelo y métricas
    """
    logger.info("Entrenando XGBoost...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    model = xgb.XGBClassifier(random_state=parameters['random_state'])
    param_grid = parameters['models']['xgboost']['param_grid']
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"XGBoost - Best params: {grid_search.best_params_}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_model_metrics(metrics, 'XGBoost')
    
    return {
        'model': best_model,
        'metrics': metrics
    }, metrics_json


def train_svc(train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de SVC con GridSearchCV.
    
    Args:
        train_data: Data de entrenamiento
        test_data: Data de test
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelo y métricas
    """
    logger.info("Entrenando SVC (usando LinearSVC para velocidad)...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    # Usar LinearSVC que es mucho más rápido que SVC con kernel RBF
    # Para obtener probabilidades, usaremos LogisticRegression con las mismas features
    model = LinearSVC(random_state=parameters['random_state'], dual=False, max_iter=1000)
    param_grid = {'C': parameters['models']['svc']['param_grid']['C']}
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Para obtener probabilidades, entrenar un LogisticRegression con las mismas features
    # y usar decision_function de LinearSVC como input
    decision_scores = best_model.decision_function(X_test)
    # Convertir decision scores a probabilidades aproximadas usando sigmoid
    y_pred_proba = 1 / (1 + np.exp(-decision_scores))
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"SVC - Best params: {grid_search.best_params_}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_model_metrics(metrics, 'SVC')
    
    return {
        'model': best_model,
        'metrics': metrics
    }, metrics_json


def train_adaboost(train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Entrenar modelo de AdaBoost con GridSearchCV.
    
    Args:
        train_data: Data de entrenamiento
        test_data: Data de test
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelo y métricas
    """
    logger.info("Entrenando AdaBoost...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    model = AdaBoostClassifier(random_state=parameters['random_state'])
    param_grid = parameters['models']['adaboost']['param_grid']
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"AdaBoost - Best params: {grid_search.best_params_}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return {
        'model': best_model,
        'metrics': metrics
    }


def train_lightgbm(train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar modelo de LightGBM con GridSearchCV.
    
    Args:
        train_data: Data de entrenamiento
        test_data: Data de test
        parameters: Parámetros de configuración
        
    Returns:
        Diccionario con modelo y métricas
    """
    logger.info("Entrenando LightGBM...")
    
    X_train, y_train = _extract_features_and_labels(train_data)
    X_test, y_test = _extract_features_and_labels(test_data)
    
    model = lgb.LGBMClassifier(random_state=parameters['random_state'], verbose=-1)
    param_grid = parameters['models']['lightgbm']['param_grid']
    cv = parameters['cv_folds']
    scoring = parameters['scoring']
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"LightGBM - Best params: {grid_search.best_params_}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_model_metrics(metrics, 'LightGBM')
    
    return {
        'model': best_model,
        'metrics': metrics
    }, metrics_json


def create_classification_comparison(
    logistic_result: Dict[str, Any],
    rf_clf_result: Dict[str, Any],
    xgb_clf_result: Dict[str, Any],
    svc_result: Dict[str, Any],
    lgbm_clf_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Crear tabla comparativa de todos los modelos de clasificación.
    
    Args:
        logistic_result: Resultados de Logistic Regression
        rf_clf_result: Resultados de Random Forest
        xgb_clf_result: Resultados de XGBoost
        svc_result: Resultados de SVC
        lgbm_clf_result: Resultados de LightGBM
        
        Returns:
        Diccionario con comparación
    """
    logger.info("Creando comparación de modelos de clasificación...")
    
    comparison = {
        'models': {
            'Logistic Regression': {
                'metrics': logistic_result['metrics'],
                'best_params': logistic_result['metrics']['best_params']
            },
            'Random Forest': {
                'metrics': rf_clf_result['metrics'],
                'best_params': rf_clf_result['metrics']['best_params']
            },
            'XGBoost': {
                'metrics': xgb_clf_result['metrics'],
                'best_params': xgb_clf_result['metrics']['best_params']
            },
            'SVC': {
                'metrics': svc_result['metrics'],
                'best_params': svc_result['metrics']['best_params']
            },
            'LightGBM': {
                'metrics': lgbm_clf_result['metrics'],
                'best_params': lgbm_clf_result['metrics']['best_params']
            }
        }
    }
    
    logger.info("Comparación de clasificación creada exitosamente")
    return comparison


def consolidate_classification_metrics(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidar métricas de clasificación en formato JSON.
    
    Args:
        comparison: Comparación de modelos
        
    Returns:
        Diccionario con métricas consolidadas
    """
    logger.info("Consolidando métricas de clasificación...")
    
    metrics_dict = {}
    
    for model_name, model_data in comparison['models'].items():
        metrics_dict[model_name] = {
            'accuracy': float(model_data['metrics']['accuracy']),
            'precision': float(model_data['metrics']['precision']),
            'recall': float(model_data['metrics']['recall']),
            'f1_score': float(model_data['metrics']['f1']),
            'roc_auc': float(model_data['metrics']['roc_auc']),
            'best_params': model_data['best_params']
        }
    
    # Determinar mejor modelo
    best_model = max(metrics_dict.items(), key=lambda x: x[1]['roc_auc'])
    metrics_dict['_best_model'] = {'name': best_model[0], 'roc_auc': best_model[1]['roc_auc']}
    
    logger.info(f"Mejor modelo de clasificación: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.4f})")
    
    return metrics_dict
