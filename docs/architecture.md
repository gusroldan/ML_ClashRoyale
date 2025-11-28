# 🏗️ Arquitectura del Proyecto ML Clash Royale

## 1. Visión General

Este proyecto implementa un pipeline completo de Machine Learning para analizar datos de combates de Clash Royale utilizando el framework **Kedro** para orquestación, **Airflow** para programación de tareas, y **DVC** para versionado de datos y modelos.

## 2. Estructura del Proyecto

```
ML_ClashRoyale/
├── airflow/
│   └── dags/                    # DAGs de Airflow
│       ├── clashroyale_ml_pipeline.py  # DAG maestro
│       ├── classification_ml_pipeline.py
│       ├── regression_ml_pipeline.py
│       ├── clustering_ml_pipeline.py
│       ├── dimensionality_reduction_ml_pipeline.py
│       └── ...
│
├── conf/
│   ├── base/
│   │   ├── catalog.yml          # Catálogo de datasets
│   │   └── parameters.yml       # Parámetros configurables
│   └── local/                   # Configuración local
│
├── data/                        # Datos organizados por fases
│   ├── 01_raw/                  # Datos originales
│   ├── 02_intermediate/         # Datos intermedios
│   ├── 03_primary/               # Datos primarios
│   ├── 04_feature/              # Features engineering
│   ├── 05_model_input/          # Datos listos para modelado
│   ├── 06_models/               # Modelos entrenados
│   ├── 07_model_output/          # Salidas de modelos
│   └── 08_reporting/            # Reportes y métricas
│
├── docs/                        # Documentación
│   ├── architecture.md          # Este archivo
│   └── unsupervised_analysis.md # Análisis no supervisado
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_final_analysis.ipynb
│
├── src/
│   └── proyecto_ml_clashroyale/
│       ├── pipelines/           # Pipelines de Kedro
│       │   ├── business_understanding.py
│       │   ├── data_preparation.py
│       │   ├── feature_engineering.py
│       │   ├── classification.py
│       │   ├── regression.py
│       │   ├── unsupervised_learning.py
│       │   ├── dimensionality_reduction.py
│       │   ├── anomaly_detection.py
│       │   ├── association_rules.py
│       │   └── nodes/          # Nodos de procesamiento
│       └── pipeline_registry.py
│
├── Dockerfile                   # Imagen Docker
├── docker-compose.yml          # Orquestación Docker
├── dvc.yaml                    # Configuración DVC
├── requirements.txt            # Dependencias Python
└── README.md                   # Documentación principal
```

## 3. Flujo de Datos

### 3.1. Pipeline Principal

```
data_engineering → supervised → unsupervised
```

**Fase 1: Data Engineering**
- `business_understanding`: Comprensión del negocio y limpieza inicial
- `data_preparation`: Preparación y unificación de datos
- `feature_engineering`: Creación de features y división train/test

**Fase 2: Supervised Learning**
- `classification`: Modelos de clasificación (5 modelos)
- `regression`: Modelos de regresión (6 modelos)

**Fase 3: Unsupervised Learning**
- `unsupervised_learning`: Clustering (K-Means, OPTICS, Hierarchical)
- `dimensionality_reduction`: PCA y UMAP
- `anomaly_detection`: Detección de anomalías (4 métodos)
- `association_rules`: Reglas de asociación (Apriori, FP-Growth)

## 4. Componentes Técnicos

### 4.1. Framework Kedro

Kedro organiza el proyecto en pipelines modulares y reutilizables:

- **Pipelines**: Conjuntos de nodos que procesan datos
- **Nodos**: Funciones que realizan transformaciones específicas
- **Catálogo**: Define dónde se guardan/cargan los datos
- **Parámetros**: Configuración centralizada en `parameters.yml`

### 4.2. Orquestación Airflow

Airflow programa y ejecuta los pipelines de forma automatizada:

- **DAG Maestro**: `clashroyale_ml_pipeline.py` orquesta todo el flujo
- **DAGs Específicos**: Cada pipeline tiene su propio DAG para ejecución independiente
- **Dependencias**: Definidas entre tareas para garantizar orden de ejecución
- **Programación**: Ejecución semanal automática del pipeline completo

### 4.3. Versionado DVC

DVC versiona datos, modelos y métricas:

- **Stages**: Cada pipeline es un stage en `dvc.yaml`
- **Dependencias**: Define qué datos necesita cada stage
- **Salidas**: Qué archivos genera cada stage
- **Métricas**: Tracking de métricas de experimentos

### 4.4. Docker

Containerización para reproducibilidad:

- **Dockerfile**: Imagen con todas las dependencias
- **docker-compose.yml**: Orquestación de servicios
- **Volúmenes**: Montaje de datos y código

## 5. Modelos Implementados

### 5.1. Supervised Learning

**Clasificación:**
- Logistic Regression
- Random Forest
- XGBoost
- LinearSVC
- LightGBM

**Regresión:**
- Linear Regression
- Ridge
- Random Forest
- XGBoost
- LinearSVR
- LightGBM

### 5.2. Unsupervised Learning

**Clustering:**
- K-Means
- OPTICS
- Hierarchical Clustering

**Reducción de Dimensionalidad:**
- PCA (Principal Component Analysis)
- UMAP (Uniform Manifold Approximation and Projection)

**Detección de Anomalías:**
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Autoencoders (opcional)

**Reglas de Asociación:**
- Apriori Algorithm
- FP-Growth

## 6. Métricas y Evaluación

### 6.1. Clasificación
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### 6.2. Regresión
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

### 6.3. Clustering
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Elbow Method (K-Means)
- Dendrograms (Hierarchical)

### 6.4. Reducción de Dimensionalidad
- Varianza Explicada (PCA)
- Loadings y Biplots (PCA)

### 6.5. Detección de Anomalías
- Número de anomalías detectadas
- Porcentaje de anomalías
- Scores de anomalía

### 6.6. Reglas de Asociación
- Support
- Confidence
- Lift

## 7. Tecnologías Utilizadas

### 7.1. Core
- **Python 3.11+**
- **Kedro 0.18+**: Framework de pipelines
- **Apache Airflow 3.1.0**: Orquestación
- **DVC 3.63+**: Versionado de datos

### 7.2. Machine Learning
- **scikit-learn 1.3.0+**: Algoritmos ML
- **XGBoost 1.7.0+**: Gradient boosting
- **LightGBM 3.3.0+**: Gradient boosting rápido
- **umap-learn 0.5.0+**: Reducción de dimensionalidad
- **pyod 1.1.0+**: Detección de anomalías
- **mlxtend 0.22.0+**: Reglas de asociación
- **hdbscan 0.8.0+**: Clustering jerárquico
- **shap 0.42.0+**: Interpretabilidad

### 7.3. Visualización
- **matplotlib 3.7.0+**
- **seaborn 0.12.0+**
- **plotly 5.0.0+**

### 7.4. Data Processing
- **pandas 1.3.0+**
- **numpy 1.21.0+**

## 8. Ejecución del Proyecto

### 8.1. Local (Kedro)
```bash
# Pipeline completo
kedro run

# Pipeline específico
kedro run --pipeline=classification
kedro run --pipeline=unsupervised_learning
```

### 8.2. Docker + Airflow
```bash
# Construir imagen
docker-compose build

# Iniciar servicios
docker-compose up -d

# Acceder a Airflow UI
# http://localhost:8080
```

### 8.3. DVC
```bash
# Ejecutar stage específico
dvc repro unsupervised_learning

# Ver métricas
dvc metrics show
```

## 9. Extensibilidad

El proyecto está diseñado para ser extensible:

- **Nuevos modelos**: Agregar nodos en `pipelines/nodes/`
- **Nuevos pipelines**: Crear archivos en `pipelines/` y registrar en `pipeline_registry.py`
- **Nuevas métricas**: Agregar cálculo en los nodos correspondientes
- **Nuevos DAGs**: Crear archivos en `airflow/dags/`

## 10. Mejores Prácticas

1. **Versionado**: Usar DVC para todos los datos y modelos
2. **Parámetros**: Centralizar configuración en `parameters.yml`
3. **Logging**: Usar el sistema de logging de Kedro
4. **Testing**: Agregar tests en `tests/`
5. **Documentación**: Mantener documentación actualizada
6. **Reproducibilidad**: Usar Docker para entornos consistentes

