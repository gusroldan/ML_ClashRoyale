"""Nodos para el pipeline de detección de anomalías."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import logging
import json

logger = logging.getLogger(__name__)

# Intentar importar tensorflow/keras para autoencoders
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow no está disponible. Los autoencoders no se podrán usar.")


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


def save_anomaly_metrics(metrics: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Guardar métricas de detección de anomalías en formato serializable.
    
    Args:
        metrics: Diccionario con métricas
        model_name: Nombre del modelo
        
    Returns:
        Diccionario con métricas convertidas a tipos básicos
    """
    result = {
        'model_name': model_name,
        'n_anomalies': int(metrics.get('n_anomalies', 0)),
        'n_normal': int(metrics.get('n_normal', 0)),
        'anomaly_percentage': float(metrics.get('anomaly_percentage', 0.0)),
        'parameters': metrics.get('parameters', {})
    }
    
    # Agregar métricas adicionales si existen
    if 'scores' in metrics:
        result['scores'] = {
            'mean': float(np.mean(metrics['scores'])),
            'std': float(np.std(metrics['scores'])),
            'min': float(np.min(metrics['scores'])),
            'max': float(np.max(metrics['scores']))
        }
    
    return result


def detect_anomalies_isolation_forest(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Detectar anomalías usando Isolation Forest.
    
    Args:
        train_data: Data de entrenamiento
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (resultado del modelo, métricas en formato JSON)
    """
    logger.info("Detectando anomalías con Isolation Forest...")
    
    X = _extract_features(train_data)
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Parámetros
    contamination = parameters['models']['isolation_forest'].get('contamination', 0.1)
    n_estimators = parameters['models']['isolation_forest'].get('n_estimators', 100)
    max_samples = parameters['models']['isolation_forest'].get('max_samples', 'auto')
    random_state = parameters.get('random_state', 42)
    
    # Entrenar Isolation Forest
    isolation_forest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1
    )
    
    predictions = isolation_forest.fit_predict(X_scaled_df)
    # Isolation Forest devuelve -1 para anomalías y 1 para normales
    anomaly_labels = (predictions == -1).astype(int)
    scores = isolation_forest.score_samples(X_scaled_df)
    
    # Calcular métricas
    n_anomalies = int(np.sum(anomaly_labels))
    n_normal = int(len(anomaly_labels) - n_anomalies)
    anomaly_percentage = float(n_anomalies / len(anomaly_labels) * 100)
    
    metrics = {
        'n_anomalies': n_anomalies,
        'n_normal': n_normal,
        'anomaly_percentage': anomaly_percentage,
        'scores': scores.tolist(),
        'parameters': {
            'contamination': contamination,
            'n_estimators': n_estimators,
            'max_samples': max_samples,
            'random_state': random_state
        }
    }
    
    logger.info(f"Isolation Forest - Anomalías detectadas: {n_anomalies} ({anomaly_percentage:.2f}%)")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_anomaly_metrics(metrics, 'Isolation Forest')
    
    return {
        'model': isolation_forest,
        'scaler': scaler,
        'predictions': predictions,
        'anomaly_labels': anomaly_labels,
        'scores': scores,
        'metrics': metrics
    }, metrics_json


def detect_anomalies_lof(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Detectar anomalías usando Local Outlier Factor (LOF).
    
    Args:
        train_data: Data de entrenamiento
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (resultado del modelo, métricas en formato JSON)
    """
    logger.info("Detectando anomalías con Local Outlier Factor (LOF)...")
    
    X = _extract_features(train_data)
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Parámetros
    n_neighbors = parameters['models']['lof'].get('n_neighbors', 20)
    contamination = parameters['models']['lof'].get('contamination', 0.1)
    metric = parameters['models']['lof'].get('metric', 'euclidean')
    
    # Entrenar LOF
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        metric=metric,
        n_jobs=-1
    )
    
    predictions = lof.fit_predict(X_scaled_df)
    # LOF devuelve -1 para anomalías y 1 para normales
    anomaly_labels = (predictions == -1).astype(int)
    scores = lof.negative_outlier_factor_
    
    # Calcular métricas
    n_anomalies = int(np.sum(anomaly_labels))
    n_normal = int(len(anomaly_labels) - n_anomalies)
    anomaly_percentage = float(n_anomalies / len(anomaly_labels) * 100)
    
    metrics = {
        'n_anomalies': n_anomalies,
        'n_normal': n_normal,
        'anomaly_percentage': anomaly_percentage,
        'scores': scores.tolist(),
        'parameters': {
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'metric': metric
        }
    }
    
    logger.info(f"LOF - Anomalías detectadas: {n_anomalies} ({anomaly_percentage:.2f}%)")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_anomaly_metrics(metrics, 'Local Outlier Factor')
    
    return {
        'model': lof,
        'scaler': scaler,
        'predictions': predictions,
        'anomaly_labels': anomaly_labels,
        'scores': scores,
        'metrics': metrics
    }, metrics_json


def detect_anomalies_oneclass_svm(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Detectar anomalías usando One-Class SVM.
    
    Args:
        train_data: Data de entrenamiento
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (resultado del modelo, métricas en formato JSON)
    """
    logger.info("Detectando anomalías con One-Class SVM...")
    
    X = _extract_features(train_data)
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Parámetros
    nu = parameters['models']['oneclass_svm'].get('nu', 0.1)
    kernel = parameters['models']['oneclass_svm'].get('kernel', 'rbf')
    gamma = parameters['models']['oneclass_svm'].get('gamma', 'scale')
    
    # Entrenar One-Class SVM
    oneclass_svm = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma=gamma
    )
    
    predictions = oneclass_svm.fit_predict(X_scaled_df)
    # One-Class SVM devuelve -1 para anomalías y 1 para normales
    anomaly_labels = (predictions == -1).astype(int)
    scores = oneclass_svm.decision_function(X_scaled_df)
    
    # Calcular métricas
    n_anomalies = int(np.sum(anomaly_labels))
    n_normal = int(len(anomaly_labels) - n_anomalies)
    anomaly_percentage = float(n_anomalies / len(anomaly_labels) * 100)
    
    metrics = {
        'n_anomalies': n_anomalies,
        'n_normal': n_normal,
        'anomaly_percentage': anomaly_percentage,
        'scores': scores.tolist(),
        'parameters': {
            'nu': nu,
            'kernel': kernel,
            'gamma': gamma
        }
    }
    
    logger.info(f"One-Class SVM - Anomalías detectadas: {n_anomalies} ({anomaly_percentage:.2f}%)")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_anomaly_metrics(metrics, 'One-Class SVM')
    
    return {
        'model': oneclass_svm,
        'scaler': scaler,
        'predictions': predictions,
        'anomaly_labels': anomaly_labels,
        'scores': scores,
        'metrics': metrics
    }, metrics_json


def detect_anomalies_autoencoder(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Detectar anomalías usando Autoencoders.
    
    Args:
        train_data: Data de entrenamiento
        parameters: Parámetros de configuración
        
    Returns:
        Tupla con (resultado del modelo, métricas en formato JSON)
    """
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow no está disponible. No se puede usar Autoencoders.")
        raise ImportError("TensorFlow es requerido para Autoencoders. Instala con: pip install tensorflow")
    
    logger.info("Detectando anomalías con Autoencoders...")
    
    X = _extract_features(train_data)
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Parámetros
    encoding_dim = parameters['models']['autoencoder'].get('encoding_dim', 32)
    hidden_layers = parameters['models']['autoencoder'].get('hidden_layers', [64, 32])
    epochs = parameters['models']['autoencoder'].get('epochs', 50)
    batch_size = parameters['models']['autoencoder'].get('batch_size', 32)
    threshold_percentile = parameters['models']['autoencoder'].get('threshold_percentile', 95)
    random_state = parameters.get('random_state', 42)
    
    input_dim = X_scaled_df.shape[1]
    
    # Construir autoencoder
    # Encoder
    encoder_input = keras.Input(shape=(input_dim,))
    x = encoder_input
    for hidden_dim in hidden_layers:
        x = layers.Dense(hidden_dim, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    
    # Decoder
    decoder_input = encoded
    x = decoder_input
    for hidden_dim in reversed(hidden_layers):
        x = layers.Dense(hidden_dim, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='sigmoid')(x)
    
    autoencoder = keras.Model(encoder_input, decoded)
    encoder = keras.Model(encoder_input, encoded)
    
    # Compilar y entrenar
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Entrenar
    history = autoencoder.fit(
        X_scaled_df, X_scaled_df,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
        validation_split=0.1
    )
    
    # Predecir y calcular errores de reconstrucción
    X_pred = autoencoder.predict(X_scaled_df, verbose=0)
    reconstruction_errors = np.mean(np.square(X_scaled_df - X_pred), axis=1)
    
    # Determinar umbral basado en percentil
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    anomaly_labels = (reconstruction_errors > threshold).astype(int)
    scores = -reconstruction_errors  # Negativo para que valores más bajos = más anómalos
    
    # Calcular métricas
    n_anomalies = int(np.sum(anomaly_labels))
    n_normal = int(len(anomaly_labels) - n_anomalies)
    anomaly_percentage = float(n_anomalies / len(anomaly_labels) * 100)
    
    metrics = {
        'n_anomalies': n_anomalies,
        'n_normal': n_normal,
        'anomaly_percentage': anomaly_percentage,
        'scores': scores.tolist(),
        'reconstruction_errors': reconstruction_errors.tolist(),
        'threshold': float(threshold),
        'parameters': {
            'encoding_dim': encoding_dim,
            'hidden_layers': hidden_layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'threshold_percentile': threshold_percentile,
            'random_state': random_state
        }
    }
    
    logger.info(f"Autoencoder - Anomalías detectadas: {n_anomalies} ({anomaly_percentage:.2f}%)")
    
    # Guardar métricas en formato JSON serializable
    metrics_json = save_anomaly_metrics(metrics, 'Autoencoder')
    
    return {
        'model': autoencoder,
        'encoder': encoder,
        'scaler': scaler,
        'anomaly_labels': anomaly_labels,
        'scores': scores,
        'reconstruction_errors': reconstruction_errors,
        'threshold': threshold,
        'metrics': metrics,
        'history': history.history
    }, metrics_json


def create_anomaly_detection_comparison(
    isolation_forest_result: Dict[str, Any],
    lof_result: Dict[str, Any],
    oneclass_svm_result: Dict[str, Any],
    autoencoder_result: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Crear comparación entre los diferentes métodos de detección de anomalías.
    
    Args:
        isolation_forest_result: Resultado de Isolation Forest
        lof_result: Resultado de LOF
        oneclass_svm_result: Resultado de One-Class SVM
        autoencoder_result: Resultado de Autoencoder (opcional)
        
    Returns:
        Diccionario con comparación de modelos
    """
    logger.info("Creando comparación de métodos de detección de anomalías...")
    
    comparison = {
        'isolation_forest': {
            'n_anomalies': int(isolation_forest_result['metrics']['n_anomalies']),
            'anomaly_percentage': float(isolation_forest_result['metrics']['anomaly_percentage']),
            'mean_score': float(np.mean(isolation_forest_result['metrics']['scores']))
        },
        'lof': {
            'n_anomalies': int(lof_result['metrics']['n_anomalies']),
            'anomaly_percentage': float(lof_result['metrics']['anomaly_percentage']),
            'mean_score': float(np.mean(lof_result['metrics']['scores']))
        },
        'oneclass_svm': {
            'n_anomalies': int(oneclass_svm_result['metrics']['n_anomalies']),
            'anomaly_percentage': float(oneclass_svm_result['metrics']['anomaly_percentage']),
            'mean_score': float(np.mean(oneclass_svm_result['metrics']['scores']))
        }
    }
    
    if autoencoder_result is not None:
        comparison['autoencoder'] = {
            'n_anomalies': int(autoencoder_result['metrics']['n_anomalies']),
            'anomaly_percentage': float(autoencoder_result['metrics']['anomaly_percentage']),
            'mean_score': float(np.mean(autoencoder_result['metrics']['scores'])),
            'threshold': float(autoencoder_result['metrics']['threshold'])
        }
    
    # Resumen
    comparison['summary'] = {
        'total_samples': int(len(isolation_forest_result['anomaly_labels'])),
        'methods_compared': list(comparison.keys()),
        'note': 'Comparación de diferentes métodos de detección de anomalías'
    }
    
    logger.info("Comparación de detección de anomalías completada")
    
    return comparison

