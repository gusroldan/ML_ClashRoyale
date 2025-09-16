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

- **Análisis Exploratorio de Datos (EDA)**: Identificar patrones en combates de Clash Royale
- **Análisis de Cartas**: Determinar las cartas más utilizadas y efectivas
- **Análisis de Win Conditions**: Evaluar la efectividad de diferentes estrategias
- **Distribución de Rarezas**: Analizar la composición de mazos por rareza
- **Preparación de Datos**: Unificar múltiples fuentes de datos para modelado

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
proyecto-ml-clashroyale/
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
│   └── 03_primary/             # Datos finales unificados
├── 📂 src/proyecto_ml_clashroyale/
│   └── pipelines/              # Pipelines de procesamiento
│       ├── business_understanding/  # Fase 1 CRISP-DM
│       ├── eda/                # Fase 2 CRISP-DM  
│       └── data_preparation/   # Fase 3 CRISP-DM
├── 📂 notebooks/               # Jupyter notebooks
├── requirements.txt            # Dependencias Python
└── README.md                   # Este archivo
```

## 🔧 Uso del Proyecto

### Ejecutar Pipeline Completo

```bash
kedro run
```

### Ejecutar Pipelines Específicos

#### Fase 1: Comprensión del Negocio
```bash
kedro run --pipeline=business_understanding
```

#### Fase 2: Análisis Exploratorio de Datos (EDA)
```bash
kedro run --pipeline=eda
```

#### Fase 3: Preparación de Datos
```bash
kedro run --pipeline=data_preparation
```

### Visualizar Pipeline (Opcional)

```bash
kedro viz
```

Abre tu navegador en `http://127.0.0.1:4141` para ver la visualización interactiva.

### Trabajar con Jupyter Notebooks

```bash
kedro jupyter notebook
```

O con JupyterLab:

```bash
kedro jupyter lab
```

## 📊 Resultados del Análisis

### EDA - Análisis Exploratorio

El pipeline de EDA genera los siguientes resultados:

- **Distribución de Rarezas**: Análisis de composición de mazos
- **Cartas Más Utilizadas**: Top 20 cartas con nombres legibles
- **Win Conditions**: Análisis de efectividad de estrategias
- **Resumen Ejecutivo**: Estadísticas consolidadas

### Preparación de Datos

- **Dataset Unificado**: 5,644,203 registros combinados
- **29 Columnas Seleccionadas**: Variables relevantes para ML
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
