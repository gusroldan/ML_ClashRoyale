# 📊 Informe Consolidado de Modelos Entrenados

Este documento describe el informe JSON consolidado que contiene todos los resultados de los modelos entrenados en el proyecto.

## 📄 Archivo Generado

El informe se guarda en:
```
data/08_reporting/models_report.json
```

## 🚀 Generación del Informe

### Opción 1: Usando Kedro Pipeline

```bash
kedro run --pipeline=reporting
```

### Opción 2: Usando Script Independiente

```bash
python scripts/generate_models_report.py
```

## 📋 Estructura del Informe

El informe JSON tiene la siguiente estructura:

```json
{
  "metadata": {
    "project_name": "ML ClashRoyale",
    "report_type": "Modelos Entrenados - Informe Consolidado",
    "generation_date": "2024-01-01T12:00:00",
    "version": "1.0"
  },
  "summary": {
    "total_models": 14,
    "supervised_models": 11,
    "unsupervised_models": 9,
    "best_models": {
      "classification": {...},
      "regression": {...},
      "clustering": {...},
      ...
    }
  },
  "supervised_learning": {
    "classification": {
      "models": {...},
      "total_models": 5,
      "task": "..."
    },
    "regression": {
      "models": {...},
      "total_models": 6,
      "task": "..."
    }
  },
  "unsupervised_learning": {
    "clustering": {...},
    "dimensionality_reduction": {...},
    "anomaly_detection": {...},
    "association_rules": {...}
  },
  "interpretability": {
    "shap_analysis": {...}
  }
}
```

## 📊 Contenido del Informe

### 1. Metadata
- Información del proyecto
- Fecha de generación
- Versión del informe

### 2. Summary
- Total de modelos entrenados
- Desglose por tipo (supervisado/no supervisado)
- Mejores modelos por tarea

### 3. Supervised Learning

#### Classification
- **Modelos incluidos**: Logistic Regression, Random Forest, XGBoost, SVC, LightGBM
- **Métricas**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Mejor modelo**: Identificado por ROC-AUC

#### Regression
- **Modelos incluidos**: Linear Regression, Ridge, Random Forest, XGBoost, SVR, LightGBM
- **Métricas**: MAE, MSE, RMSE, R²
- **Mejor modelo**: Identificado por menor MAE

### 4. Unsupervised Learning

#### Clustering
- **Algoritmos**: K-Means, OPTICS, Hierarchical
- **Métricas**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- **Mejor algoritmo**: Identificado por Silhouette Score

#### Dimensionality Reduction
- **PCA**: Varianza explicada, número de componentes
- **UMAP**: Parámetros de configuración

#### Anomaly Detection
- **Algoritmos**: Isolation Forest, LOF, One-Class SVM, Autoencoders
- **Métricas**: Número de anomalías, porcentaje, scores

#### Association Rules
- **Algoritmos**: Apriori, FP-Growth
- **Métricas**: Itemsets frecuentes, reglas generadas, lift promedio

### 5. Interpretability
- **SHAP Analysis**: Resúmenes de interpretabilidad para modelos de clasificación y regresión
- **Top Features**: Features más importantes según SHAP

## 🔍 Uso del Informe

### Cargar en Python

```python
import json

with open('data/08_reporting/models_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

# Acceder a información específica
print(f"Total de modelos: {report['summary']['total_models']}")
print(f"Mejor modelo de clasificación: {report['summary']['best_models']['classification']['model']}")
```

### Visualización

El informe puede usarse para:
- Generar dashboards interactivos
- Crear reportes HTML/PDF
- Comparar experimentos
- Documentar resultados del proyecto

## 📝 Notas

- El informe se genera automáticamente cuando se ejecutan los pipelines
- Si algún modelo no está disponible, el informe se genera con los modelos disponibles
- Los campos opcionales pueden ser `null` si no están disponibles

