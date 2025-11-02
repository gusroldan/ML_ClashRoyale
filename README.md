# 🏰 Proyecto ML ClashRoyale

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 🎥 Video Explicativo

📺 **[Ver Video del Proyecto](https://drive.google.com/file/d/12JK-j3v5s3S2cdgWdI1WtRYYEJT-NT2h/view)** - Explicación completa del proyecto, metodología CRISP-DM y resultados obtenidos.

## 👥 Autores

- **Nicolás Hernández** - *Desarrollo y Análisis* - [GitHub](https://github.com/NicolasBeatum)
- **Gustavo Roldán** - *Desarrollo y Análisis* - [GitHub](https://github.com/gusroldan)

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline de Machine Learning para analizar datos de combates de Clash Royale utilizando el framework **Kedro**. El proyecto sigue la metodología **CRISP-DM** y está diseñado para procesar y analizar millones de registros de batallas para identificar patrones, cartas más efectivas y estrategias ganadoras.

### 🎯 Objetivos del Proyecto

#### Análisis Exploratorio (CRISP-DM)
- **Análisis Exploratorio de Datos (EDA)**: Identificar patrones en combates de Clash Royale
- **Análisis de Cartas**: Determinar las cartas más utilizadas y efectivas
- **Análisis de Win Conditions**: Evaluar la efectividad de diferentes estrategias
- **Distribución de Rarezas**: Analizar la composición de mazos por rareza
- **Preparación de Datos**: Unificar múltiples fuentes de datos para modelado

#### Machine Learning - Modelos Entrenados

**Clasificación (5 modelos)**:
- **Objetivo**: Predecir si el jugador A gana o pierde la batalla (binario 1/0)
- **Modelos**: Logistic Regression, Random Forest, XGBoost, LinearSVC, LightGBM
- **Métricas**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Técnica**: GridSearchCV con Cross-Validation (k≥3 folds)

**Regresión (6 modelos)**:
- **Objetivo**: Predecir el cambio de trofeos del jugador A (A.trophyChange, con signo)
- **Modelos**: Linear Regression, Ridge, Random Forest, XGBoost, LinearSVR, LightGBM
- **Métricas**: MAE, MSE, RMSE, R²
- **Técnica**: GridSearchCV con Cross-Validation (k≥3 folds)

**Feature Engineering**:
- Re-etiquetado balanceado (50% A=winner, 50% A=loser)
- Multi-hot encoding de 102 cartas (cards_diff = vec_A - vec_B)
- Features agregadas (Δtrophies, conteos de rareza)
- División train/test para validación

### 📊 Datasets Incluidos

Este proyecto utiliza el dataset **Clash Royale Season 18** disponible en Kaggle:

🔗 **Fuente de Datos**: [Clash Royale Season 18 (Dec 03/20) Dataset](https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset)

**Archivos utilizados en `data/01_raw/`:**

- **3 Datasets de Combates**: 
  - `battlesStaging_12272020_WL_tagged.csv` - Combates del 27/12/2020
  - `BattlesStaging_12312020_WL_tagged.csv` - Combates del 31/12/2020  
  - `BattlesStaging_01042021_WL_tagged.csv` - Combates del 04/01/2021
- **Lista Maestra de Cartas**: `CardMasterListSeason18_12082020.csv` - Catálogo completo de 102 cartas disponibles
- **Win Conditions**: `Wincons.csv` - 24 condiciones de victoria identificadas

**Total de registros**: Más de 5.6 millones de batallas analizadas

## 🚀 Instalación Rápida

### Prerrequisitos

- Python 3.8 o superior
- Git
- 8GB RAM mínimo (recomendado para procesar datasets grandes)

### 1. Obtener los Datos

**Opción A: Descargar desde Kaggle**
1. Visita el dataset: [Clash Royale Season 18 Dataset](https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset)
2. Descarga los siguientes archivos a la carpeta `data/01_raw/`:
   - `battlesStaging_12272020_WL_tagged.csv`
   - `BattlesStaging_12312020_WL_tagged.csv`
   - `BattlesStaging_01042021_WL_tagged.csv`
   - `CardMasterListSeason18_12082020.csv`
   - `Wincons.csv`


### 2. Crear Entorno Virtual

#### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Windows (CMD):
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar Instalación

```bash
kedro info
```

Deberías ver la información de Kedro v1.0.0 con los plugins instalados.

## 📁 Estructura del Proyecto

```
ML_ClashRoyale/
├── 📂 conf/                    # Configuraciones
│   ├── base/                   # Configuración base
│   │   ├── catalog.yml         # Catálogo de datasets
│   │   └── parameters.yml      # Parámetros del proyecto
│   └── local/                  # Configuración local
├── 📂 data/                    # Datos organizados por fases
│   ├── 01_raw/                 # Datos originales de Kaggle (5 archivos)
│   │   ├── battlesStaging_12272020_WL_tagged.csv
│   │   ├── BattlesStaging_12312020_WL_tagged.csv
│   │   ├── BattlesStaging_01042021_WL_tagged.csv
│   │   ├── CardMasterListSeason18_12082020.csv
│   │   └── Wincons.csv
│   ├── 02_intermediate/        # Datos procesados
│   ├── 03_primary/             # Datos finales unificados
│   ├── 04_feature/             # Features para ML
│   ├── 05_model_input/          # Datos de entrenamiento/test
│   ├── 06_models/              # Modelos entrenados
│   │   ├── classification/     # Modelos de clasificación
│   │   └── regression/         # Modelos de regresión
│   ├── 07_model_output/        # Métricas por modelo (JSON)
│   └── 08_reporting/           # Comparaciones y reportes
├── 📂 src/proyecto_ml_clashroyale/
│   └── pipelines/              # Pipelines de procesamiento
│       ├── business_understanding/  # Fase 1 CRISP-DM
│       ├── eda/                # Fase 2 CRISP-DM  
│       ├── data_preparation/   # Fase 3 CRISP-DM
│       ├── feature_engineering/ # Ingeniería de features
│       ├── classification/     # Pipeline de clasificación
│       └── regression/         # Pipeline de regresión
├── 📂 airflow/                 # Configuración de Airflow
│   ├── dags/                   # DAGs de Airflow (5 DAGs)
│   ├── logs/                   # Logs de ejecución
│   └── airflow.cfg             # Configuración de Airflow
├── 📂 notebooks/               # Jupyter notebooks
├── Dockerfile                  # Imagen Docker
├── docker-compose.yml          # Configuración Docker Compose
├── docker-entrypoint.sh        # Script de inicio Docker
├── dvc.yaml                    # Configuración DVC
├── requirements.txt           # Dependencias Python
└── README.md                  # Este archivo
```

## 🚀 Ejecución del Proyecto

El proyecto se puede ejecutar de dos formas:

### Opción 1: Docker (Recomendado) 🐳

**Ventajas**: Entorno completo preconfigurado, no requiere instalación local, funciona en Windows sin WSL.

#### Prerrequisitos
- Docker Desktop instalado y ejecutándose
- Docker Compose (incluido en Docker Desktop)

#### Pasos de instalación y ejecución

**Paso 1: Construir la imagen Docker**
```bash
# IMPORTANTE: Ejecutar desde la raíz del proyecto (donde está docker-compose.yml)
# Verificar que estás en el directorio correcto:
ls docker-compose.yml pyproject.toml requirements.txt

# Construir la imagen
docker-compose build
```
⏱️ **Tiempo**: 5-15 minutos (primera vez)  
📦 **Tamaño**: ~3.4 GB (solo código + dependencias, datos se montan como volumen)

**⚠️ Nota importante sobre volúmenes:**
El `docker-compose.yml` monta el código del proyecto como volumen (`.:/app:rw`). Esto significa:
- **Ventaja**: Los cambios en el código se reflejan inmediatamente sin reconstruir la imagen
- **Requisito**: El proyecto **debe estar completo** en el host donde ejecutas `docker-compose`
- **Archivos necesarios**: `pyproject.toml`, `requirements.txt`, `src/`, `conf/`, `Dockerfile`, etc.
- **Si el proyecto no está completo en el host**, el volumen montará un directorio incompleto y Kedro no encontrará el proyecto

**Paso 2: Iniciar Airflow**
```bash
docker-compose up -d
```

**Paso 3: Acceder a la interfaz web**
- URL: http://localhost:8080
- Usuario: `admin`
- Contraseña: `admin`

**Paso 4: Ver logs (opcional)**
```bash
docker-compose logs -f airflow
```

**Paso 5: Detener Airflow**
```bash
docker-compose down
```

#### Comandos útiles de Docker
```bash
# Ver logs en tiempo real
docker-compose logs -f airflow

# Detener y limpiar volúmenes (reset completo de BD)
docker-compose down -v

# Reconstruir imagen después de cambios
docker-compose build --no-cache
docker-compose up -d

# Ejecutar comandos dentro del contenedor
docker-compose exec airflow bash

# Ver estado del contenedor
docker-compose ps
```

#### Solución de Problemas Comunes

**Error: `exec /docker-entrypoint.sh: no such file or directory`**

Este error suele ocurrir cuando el archivo `docker-entrypoint.sh` tiene finales de línea de Windows (CRLF) en lugar de Unix (LF).

**Solución 1: Convertir finales de línea (Recomendado)**
```bash
# En Git Bash o WSL
dos2unix docker-entrypoint.sh
# O usando sed
sed -i 's/\r$//' docker-entrypoint.sh

# Luego reconstruir la imagen
docker-compose build --no-cache
docker-compose up -d
```

**Solución 2: Usar editor que preserve formato Unix**
- En VS Code: Cambiar "CRLF" a "LF" en la barra de estado (clic derecho → "Change End of Line Sequence")
- Guardar el archivo `docker-entrypoint.sh` con finales de línea LF
- Reconstruir: `docker-compose build --no-cache && docker-compose up -d`

**Solución 3: Verificar que el archivo existe**
```bash
# Verificar que el archivo está en el directorio raíz
ls -la docker-entrypoint.sh

# Verificar permisos
chmod +x docker-entrypoint.sh

# Reconstruir desde cero
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

**Error: `Bash command failed. The command returned a non-zero exit code 2` (al ejecutar DAGs)**

Este error puede tener varias causas:

**Causa 1: Directorio del proyecto no existe o no es accesible**
```bash
# Verificar que el contenedor tiene el código montado correctamente
docker-compose exec airflow ls -la /app

# Verificar que pyproject.toml existe (necesario para que Kedro reconozca el proyecto)
docker-compose exec airflow test -f /app/pyproject.toml && echo "pyproject.toml existe" || echo "pyproject.toml NO existe"

# Verificar estructura del proyecto
docker-compose exec airflow ls -la /app/src
docker-compose exec airflow ls -la /app/conf

# Verificar que el volumen se montó correctamente desde el host
docker-compose exec airflow pwd
```

**Causa 2: Kedro no está instalado o no funciona**
```bash
# Verificar instalación de Kedro dentro del contenedor
docker-compose exec airflow python -m kedro --version

# Verificar que Kedro reconoce el proyecto
docker-compose exec airflow bash -c "cd /app && python -m kedro info"

# Si falla "Kedro project not found", verificar que todos los archivos están presentes
docker-compose exec airflow bash -c "cd /app && ls -la pyproject.toml conf/base src/"

# Si falla, reinstalar dependencias
docker-compose exec airflow pip install -r /app/requirements.txt
```

**Nota importante**: El proyecto debe estar completo en el host para que el volumen funcione. Asegúrate de que tienes:
- `pyproject.toml` en la raíz
- `conf/` con subdirectorios `base/` y `local/`
- `src/proyecto_ml_clashroyale/` con el código
- `requirements.txt`

**Causa 3: Datos faltantes o archivos intermedios no generados**

Este error ocurre cuando un pipeline intenta leer archivos que deberían haberse generado por otro pipeline anterior.

```bash
# Verificar que los datos de entrada existen
docker-compose exec airflow ls -la /app/data/01_raw/

# Verificar que los archivos intermedios existen (si son necesarios)
docker-compose exec airflow ls -la /app/data/02_intermediate/

# Si faltan archivos intermedios, ejecutar los pipelines en orden:
# 1. business_understanding (genera Combates*_cleaned.csv)
# 2. data_preparation (necesita Combates*_cleaned.csv)
# 3. feature_engineering
# 4. classification / regression
```

**Error específico: `DatasetError: Failed while loading data from dataset`**

Si ves este error mencionando archivos como `Combates1_cleaned.csv`, significa que:
- El pipeline `business_understanding` debe ejecutarse primero para generar estos archivos
- El DAG `clashroyale_ml_with_datacleaning` ahora incluye `business_understanding` automáticamente
- Si usas un DAG que solo ejecuta `data_preparation`, asegúrate de que los archivos `Combates*_cleaned.csv` ya existen en `data/02_intermediate/`

**Causa 4: Permisos incorrectos**
```bash
# Verificar permisos del directorio
docker-compose exec airflow ls -la /app

# Ajustar permisos si es necesario (dentro del contenedor)
docker-compose exec airflow chown -R airflow:airflow /app
```

**Solución general: Ver logs detallados**
1. En la interfaz de Airflow, haz clic en la tarea que falló
2. Haz clic en "Log" para ver el error completo
3. Revisa los mensajes de error específicos (ej: "No such file", "command not found", etc.)

**Solución: Reconstruir y verificar**
```bash
# 1. VERIFICAR que estás en el directorio correcto del proyecto
pwd  # Debe ser la raíz del proyecto
ls -la pyproject.toml requirements.txt src/ conf/  # Deben existir

# 2. Si el proyecto está incompleto, clonarlo o descargarlo completo
# Asegúrate de tener TODOS los archivos del proyecto en el host

# 3. Limpiar todo
docker-compose down -v

# 4. Reconstruir imagen
docker-compose build --no-cache

# 5. Iniciar y verificar
docker-compose up -d

# 6. Verificar que el proyecto está montado correctamente
docker-compose exec airflow ls -la /app/pyproject.toml
docker-compose exec airflow bash -c "cd /app && python -m kedro info"
```

**Si sigue fallando: Verificar el montaje del volumen**
```bash
# Verificar desde dónde se está montando el volumen
docker-compose config | grep -A 5 volumes

# Verificar que el directorio en el host tiene todos los archivos
ls -la | grep -E "pyproject.toml|requirements.txt|src|conf|Dockerfile"

# Si faltan archivos, el problema es que el proyecto no está completo en el host
```

**Otros errores comunes**
- **Error de permisos**: Verificar que `docker-entrypoint.sh` tiene permisos de ejecución
- **Error "command not found"**: Verificar que el archivo se copió correctamente en el Dockerfile
- **Error de construcción**: Limpiar cache: `docker system prune -a` y reconstruir

#### DAGs disponibles en Docker

Ver sección **"📋 DAGs de Airflow"** más abajo para detalles completos.

**Nota importante**: La base de datos de Airflow se limpia y se recrea automáticamente cada vez que inicias el contenedor. El usuario `admin` con contraseña `admin` se crea automáticamente en cada inicio.

**📖 Guía completa:** Ver [`DOCKER_GUIDE.md`](DOCKER_GUIDE.md) para instrucciones detalladas paso a paso y solución de problemas.

---

### Opción 2: Kedro (Local - Sin Docker) 🔧

**Ventajas**: Mayor control del entorno, ejecución directa, útil para desarrollo.

#### Prerrequisitos
- Python 3.8 o superior
- Entorno virtual creado
- Dependencias instaladas (ver sección "Instalación Rápida")

#### Ejecutar pipelines completos

**Pipeline completo (todos los pipelines)**
```bash
kedro run
```

**Pipelines de análisis (CRISP-DM)**
```bash
# Fase 1: Comprensión del Negocio
kedro run --pipeline=business_understanding

# Fase 2: Análisis Exploratorio de Datos (EDA)
kedro run --pipeline=eda

# Fase 3: Preparación de Datos
kedro run --pipeline=data_preparation
```

**Pipelines de Machine Learning**
```bash
# Feature Engineering (requerido antes de ML)
kedro run --pipeline=feature_engineering

# Pipeline de Clasificación (entrena 5 modelos)
kedro run --pipeline=classification

# Pipeline de Regresión (entrena 6 modelos)
kedro run --pipeline=regression
```

#### Ejecutar modelos individuales por tags

**Clasificación** (5 modelos disponibles)
```bash
kedro run --pipeline=classification --tags logistic
kedro run --pipeline=classification --tags random_forest
kedro run --pipeline=classification --tags xgboost
kedro run --pipeline=classification --tags svc
kedro run --pipeline=classification --tags lightgbm
```

**Regresión** (6 modelos disponibles)
```bash
kedro run --pipeline=regression --tags linear
kedro run --pipeline=regression --tags ridge
kedro run --pipeline=regression --tags random_forest
kedro run --pipeline=regression --tags xgboost
kedro run --pipeline=regression --tags svr
kedro run --pipeline=regression --tags lightgbm
```

#### Herramientas adicionales de Kedro

**Visualizar pipeline**
```bash
kedro viz
```
Abre `http://127.0.0.1:4141` para ver la visualización interactiva del pipeline.

**Jupyter Notebooks**
```bash
# Jupyter Notebook
kedro jupyter notebook

# JupyterLab
kedro jupyter lab
```

---

## 🧩 Pipelines ML Disponibles

### Feature Engineering (`feature_engineering`)
- **Objetivo**: Preparar datos para modelado ML
- **Tareas**:
  - Re-etiquetado balanceado (50% A=winner, 50% A=loser)
  - Multi-hot encoding de cartas (102 IDs)
  - Features agregadas (Δtrophies, conteos de rareza)
  - División train/test
- **Salidas**:
  - `data/04_feature/balanced_dataset.csv`
  - `data/04_feature/features_combined.csv`
  - `data/05_model_input/train_data.csv`, `test_data.csv`

### Clasificación (`classification`)
- **Objetivo**: Predecir si el jugador A gana (binario 1/0)
- **Modelos**: 5 modelos con GridSearchCV (k=3 folds)
  1. **Logistic Regression**: Modelo lineal básico
  2. **Random Forest**: Ensemble con árboles
  3. **XGBoost**: Gradient boosting optimizado
  4. **LinearSVC**: Support Vector Classifier lineal
  5. **LightGBM**: Gradient boosting rápido
- **Métricas**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Salidas**:
  - Modelos: `data/06_models/classification/*.pkl`
  - Métricas: `data/07_model_output/*_metrics.json`
  - Comparación: `data/08_reporting/classification_comparison.json`

### Regresión (`regression`)
- **Objetivo**: Predecir el cambio de trofeos de A (A.trophyChange, con signo)
- **Modelos**: 6 modelos con GridSearchCV (k=3 folds)
  1. **Linear Regression**: Modelo lineal básico
  2. **Ridge**: Regresión con regularización L2
  3. **Random Forest**: Ensemble con árboles
  4. **XGBoost**: Gradient boosting optimizado
  5. **LinearSVR**: Support Vector Regressor lineal
  6. **LightGBM**: Gradient boosting rápido
- **Métricas**: MAE, MSE, RMSE, R²
- **Salidas**:
  - Modelos: `data/06_models/regression/*.pkl`
  - Métricas: `data/07_model_output/*_metrics.json`
  - Comparación: `data/08_reporting/regression_comparison.json`

---

## 📋 DAGs de Airflow

El proyecto incluye 5 DAGs configurados para ejecutar los pipelines de Kedro:

### 1. `clashroyale_ml_with_datacleaning` (Completo)
- **Descripción**: Pipeline ML completo desde la limpieza de datos hasta el entrenamiento
- **Tareas**:
  1. `business_understanding` → Limpieza inicial de datos y comprensión del negocio (genera `Combates*_cleaned.csv`)
  2. `data_cleaning` → Preparación de datos y combinación de datasets
  3. `feature_engineering` → Ingeniería de features
  4. `classification_pipeline` → Entrenamiento de modelos de clasificación
  5. `regression_pipeline` → Entrenamiento de modelos de regresión
- **Flujo**: `business_understanding → data_cleaning → feature_engineering → [classification, regression]`
- **Uso**: Ejecutar el pipeline completo desde cero (requiere datos en `data/01_raw/`)

### 2. `clashroyale_ml_no_datacleaning` (Sin data cleaning)
- **Descripción**: Pipeline ML sin la etapa de limpieza de datos (asume datos ya limpios)
- **Tareas**:
  1. `feature_engineering` → Ingeniería de features
  2. `classification_pipeline` → Entrenamiento de modelos de clasificación
  3. `regression_pipeline` → Entrenamiento de modelos de regresión
- **Flujo**: `feature_engineering → [classification, regression]`
- **Uso**: Cuando los datos ya están limpios y solo necesitas re-entrenar modelos

### 3. `classification_ml_pipeline` (Solo clasificación)
- **Descripción**: Pipeline completo para entrenar modelos de clasificación
- **Tareas**:
  1. `feature_engineering` → Ingeniería de features
  2. `classification_pipeline` → Entrenamiento de 5 modelos de clasificación
- **Flujo**: `feature_engineering → classification`
- **Uso**: Cuando solo necesitas entrenar modelos de clasificación

### 4. `regression_ml_pipeline` (Solo regresión)
- **Descripción**: Pipeline completo para entrenar modelos de regresión
- **Tareas**:
  1. `feature_engineering` → Ingeniería de features
  2. `regression_pipeline` → Entrenamiento de 6 modelos de regresión
- **Flujo**: `feature_engineering → regression`
- **Uso**: Cuando solo necesitas entrenar modelos de regresión

### 5. `data_cleaning_only` (Solo limpieza)
- **Descripción**: Solo ejecuta la limpieza y preparación de datos
- **Tareas**:
  1. `data_cleaning` → Limpieza y preparación de datos
- **Flujo**: Solo `data_cleaning`
- **Uso**: Cuando solo necesitas limpiar los datos sin entrenar modelos

### Ejecutar DAGs desde Airflow

1. Accede a http://localhost:8080
2. Inicia sesión con `admin` / `admin`
3. En la lista de DAGs, selecciona el DAG deseado
4. Haz clic en "Trigger DAG" para ejecutarlo manualmente
5. Monitorea el progreso en la vista de árbol o gráfico
6. Revisa los logs de cada tarea si es necesario

### Configuración de DAGs

- **Owner**: `clashroyale_ml`
- **Retries**: 1 intento
- **Retry delay**: 5 minutos
- **Rutas**: 
  - En Docker: `/app` (contenedor)
  - En WSL: `/mnt/c/Users/Usuario/Documents/GitHub/ML_ClashRoyale` (WSL)

## 🛠️ Parámetros clave (conf/base/parameters.yml)

### Feature Engineering
- `feature_engineering.max_samples_per_class`: Tamaño por clase para balanceo (p.ej. 50,000 → 100K total)
  - Reduce memoria y tiempo de ejecución
  - Ajustable según recursos disponibles

### Clasificación
- `classification.cv_folds`: Folds de cross-validation para GridSearchCV (p.ej. 3)
- `classification.scoring`: Métricas para evaluación
- `classification.random_state`: Semilla para reproducibilidad
- `classification.param_grid`: Grid de hiperparámetros por modelo (reducido para eficiencia)

### Regresión
- `regression.cv_folds`: Folds de cross-validation para GridSearchCV (p.ej. 3)
- `regression.scoring`: Métricas para evaluación
- `regression.random_state`: Semilla para reproducibilidad
- `regression.param_grid`: Grid de hiperparámetros por modelo (reducido para eficiencia)

**Nota**: Los grids de hiperparámetros están reducidos para un uso eficiente de recursos. Ajusta según tus necesidades.


## 🎯 Objetivos de Modelado

### Clasificación (Binaria)
- **Variable objetivo**: Predecir si el jugador A gana o pierde la batalla
- **Label**: 
  - `1` = A gana
  - `0` = A pierde
- **Métricas evaluadas**: 
  - Accuracy (Precisión global)
  - Precision (Precisión por clase)
  - Recall (Sensibilidad)
  - F1-Score (Media armónica)
  - ROC-AUC (Área bajo la curva ROC)
- **Modelos**: 5 modelos con GridSearchCV + Cross-Validation (k≥3)

### Regresión
- **Variable objetivo**: Predecir el cambio de trofeos del jugador A (A.trophyChange, con signo)
  - Valores positivos = A gana trofeos
  - Valores negativos = A pierde trofeos
- **Re-etiquetado**: 
  - 50% A=winner → `y_reg = winner.trophyChange`
  - 50% A=loser → `y_reg = loser.trophyChange`
- **Métricas evaluadas**: 
  - MAE (Error Absoluto Medio)
  - MSE (Error Cuadrático Medio)
  - RMSE (Raíz del Error Cuadrático Medio)
  - R² (Coeficiente de Determinación)
- **Modelos**: 6 modelos con GridSearchCV + Cross-Validation (k≥3)

## ⚙️ Configuración Detallada

### Catálogo (conf/base/catalog.yml)
- **Modelos entrenados**: `data/06_models/{classification|regression}/*.pkl`
- **Métricas individuales**: `data/07_model_output/*_metrics.json`
- **Comparaciones**: `data/08_reporting/{classification|regression}_comparison.json`
- **Features**: `data/04_feature/*.csv`, `data/05_model_input/{train|test}_data.csv`

### Estructura de Features

**Multi-hot encoding de cartas**:
- Diccionario de 102 cartas (`card_index`)
- Vectores de presencia para A y B
- Feature principal: `cards_diff = vec_A - vec_B`

**Features agregadas**:
- `Δtrophies = A.startingTrophies - B.startingTrophies`
- Conteos de rareza (common/rare/epic/legendary) para A y B
- Variables adicionales según necesidades

**Nota importante**: `winner.trophyChange` y `loser.trophyChange` NO se usan como features, solo como target en regresión.

### Requisitos del Sistema
- **RAM**: 8GB mínimo (16GB recomendado para datasets completos)
- **Almacenamiento**: 2GB libres
- **CPU**: Multi-core recomendado
- **Docker** (opcional): Docker Desktop para ejecución con contenedores

## 🔄 DVC (Versionado de Datos y Artefactos)

El proyecto incluye DVC para versionar datasets, features, modelos y métricas.

### Inicialización (si aún no se hizo)
```bash
dvc init
git add .dvc .dvcignore
git commit -m "chore(dvc): init"
```

### Ejecución orquestada
```bash
# Ejecutar stages definidos en dvc.yaml
dvc repro

# Ejecutar un stage específico
dvc repro feature_engineering
dvc repro classification
dvc repro regression
```

### Métricas y comparación
```bash
# Ver métricas actuales
dvc metrics show

# Comparar métricas entre commits
dvc metrics diff

# Ver diferencias visualmente
dvc metrics diff --show-json
```

### Versionado de artefactos
Los siguientes directorios pueden versionarse con DVC:
- `data/04_feature/`: Features generadas
- `data/06_models/`: Modelos entrenados
- `data/07_model_output/`: Métricas de modelos
- `data/08_reporting/`: Comparaciones y reportes

**Sugerencia**: Versionar modelos y métricas permite comparar experimentos entre commits.

## 📊 Resultados del Análisis

### EDA - Análisis Exploratorio

El pipeline de EDA genera los siguientes resultados:

- **Distribución de Rarezas**: Análisis de composición de mazos
- **Cartas Más Utilizadas**: Top 20 cartas con nombres legibles
- **Win Conditions**: Análisis de efectividad de estrategias
- **Resumen Ejecutivo**: Estadísticas consolidadas

### Preparación de Datos

- **Dataset Unificado**: 5,644,203 registros combinados
- **34 Columnas Seleccionadas**: Variables relevantes para ML
- **Validación Completa**: 0 duplicados, 0 valores faltantes

## 🛠️ Configuración Avanzada

### Parámetros del Proyecto

Edita `conf/base/parameters.yml` para ajustar:

```yaml
# Ejemplo de parámetros personalizables
eda_params:
  top_cards_limit: 20
  min_usage_threshold: 0.01

data_preparation:
  selected_columns:
    - battle_id
    - winner.tag
    - loser.tag
    # ... más columnas
```

### Catálogo de Datos

El archivo `conf/base/catalog.yml` define todos los datasets del proyecto:

```yaml
# Datasets de combates del dataset de Kaggle
Combates1:
  type: kedro_datasets.pandas.CSVDataset
  filepath: data/01_raw/battlesStaging_12272020_WL_tagged.csv
  load_args:
    encoding: utf-8
    low_memory: false

Combates2:
  type: kedro_datasets.pandas.CSVDataset
  filepath: data/01_raw/BattlesStaging_12312020_WL_tagged.csv
  load_args:
    encoding: utf-8
    low_memory: false

Combates3:
  type: kedro_datasets.pandas.CSVDataset
  filepath: data/01_raw/BattlesStaging_01042021_WL_tagged.csv
  load_args:
    encoding: utf-8
    low_memory: false

# Datasets de referencia
card_master_list:
  type: kedro_datasets.pandas.CSVDataset
  filepath: data/01_raw/CardMasterListSeason18_12082020.csv
  load_args:
    encoding: utf-8

wincons:
  type: kedro_datasets.pandas.CSVDataset
  filepath: data/01_raw/Wincons.csv
  load_args:
    encoding: utf-8
```


## 📈 Rendimiento y Recursos

### Requisitos del Sistema

- **RAM**: 8GB mínimo (16GB recomendado)
- **Almacenamiento**: 2GB libres
- **CPU**: Multi-core recomendado para procesamiento paralelo

### Tiempos de Ejecución Estimados

- **Pipeline Completo**: 15-30 minutos
- **EDA**: 5-10 minutos
- **Preparación de Datos**: 10-20 minutos


## 📚 Recursos Adicionales

- [Documentación de Kedro](https://docs.kedro.org)
- [Metodología CRISP-DM](https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn](https://scikit-learn.org/)


## 🙏 Agradecimientos

- **Framework Kedro** por la excelente arquitectura y herramientas de pipeline
- **bwandowando** por proporcionar el dataset [Clash Royale Season 18](https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset) en Kaggle
- **Supercell** y la comunidad de Clash Royale por los datos del juego
---

**¡Disfruta analizando datos de Clash Royale! 🏰⚔️**
