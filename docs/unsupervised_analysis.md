# 🔍 Análisis de Aprendizaje No Supervisado

## 1. Introducción

Este documento describe los métodos de aprendizaje no supervisado implementados en el proyecto ML Clash Royale. El aprendizaje no supervisado nos permite descubrir patrones ocultos en los datos sin necesidad de etiquetas.

## 2. Clustering

### 2.1. Objetivo

Identificar grupos naturales de batallas similares basándose en las características de los mazos y jugadores.

### 2.2. Algoritmos Implementados

#### K-Means
- **Tipo**: Clustering particional basado en centroides
- **Parámetros**:
  - `n_clusters`: Número de clusters (default: 5)
  - `max_iter`: Máximo de iteraciones (default: 300)
  - `n_init`: Número de inicializaciones (default: 10)
- **Ventajas**: Rápido, escalable, fácil de interpretar
- **Desventajas**: Requiere especificar número de clusters, asume clusters esféricos

#### OPTICS
- **Tipo**: Clustering basado en densidad
- **Parámetros**:
  - `min_samples`: Mínimo de muestras en un cluster (default: 5)
  - `max_eps`: Distancia máxima (default: sin límite)
  - `metric`: Métrica de distancia (default: 'euclidean')
- **Ventajas**: No requiere número de clusters, detecta clusters de forma arbitraria
- **Desventajas**: Más lento que K-Means, puede identificar puntos de ruido

#### Hierarchical Clustering
- **Tipo**: Clustering jerárquico aglomerativo
- **Parámetros**:
  - `n_clusters`: Número de clusters (default: 5)
  - `linkage`: Método de linkage ('ward', 'complete', 'average', 'single')
  - `metric`: Métrica de distancia
- **Ventajas**: Genera dendrogramas, no requiere número inicial de clusters
- **Desventajas**: Computacionalmente costoso para datasets grandes

### 2.3. Métricas de Evaluación

- **Silhouette Score**: Mide qué tan bien separados están los clusters (rango: -1 a 1, mayor es mejor)
- **Davies-Bouldin Index**: Mide la separación entre clusters (menor es mejor)
- **Calinski-Harabasz Index**: Ratio de varianza entre y dentro de clusters (mayor es mejor)
- **Elbow Method**: Para K-Means, ayuda a determinar número óptimo de clusters
- **Dendrograms**: Para Hierarchical, visualiza la jerarquía de clusters

### 2.4. Interpretación de Resultados

Los resultados de clustering pueden revelar:
- Tipos de mazos dominantes
- Estrategias de juego similares
- Patrones de uso de cartas por grupo

## 3. Reducción de Dimensionalidad

### 3.1. Objetivo

Reducir el número de características manteniendo la información más importante, facilitando visualización y análisis.

### 3.2. Técnicas Implementadas

#### PCA (Principal Component Analysis)
- **Tipo**: Transformación lineal basada en varianza
- **Parámetros**:
  - `n_components`: Número de componentes o fracción de varianza (default: 95%)
  - `variance_threshold`: Varianza mínima a explicar (default: 0.95)
- **Ventajas**: Interpretable, preserva varianza, rápido
- **Desventajas**: Asume relaciones lineales

**Métricas**:
- Varianza explicada por componente
- Varianza acumulada
- Loadings (contribución de cada feature)
- Biplots (visualización de componentes y features)

#### UMAP (Uniform Manifold Approximation and Projection)
- **Tipo**: Reducción no lineal basada en variedades
- **Parámetros**:
  - `n_components`: Dimensiones de salida (default: 2)
  - `n_neighbors`: Número de vecinos (default: 15)
  - `min_dist`: Distancia mínima entre puntos (default: 0.1)
  - `metric`: Métrica de distancia (default: 'euclidean')
- **Ventajas**: Preserva estructura local y global, mejor para datos no lineales
- **Desventajas**: Menos interpretable que PCA, más lento

### 3.3. Aplicaciones

- **Visualización**: Proyectar datos de alta dimensionalidad a 2D/3D
- **Preprocesamiento**: Reducir dimensionalidad antes de otros algoritmos
- **Análisis exploratorio**: Identificar patrones visuales

## 4. Detección de Anomalías

### 4.1. Objetivo

Identificar batallas o patrones inusuales que se desvían significativamente del comportamiento normal.

### 4.2. Métodos Implementados

#### Isolation Forest
- **Tipo**: Basado en árboles de decisión
- **Parámetros**:
  - `contamination`: Proporción esperada de anomalías (default: 0.1)
  - `n_estimators`: Número de árboles (default: 100)
  - `max_samples`: Muestras por árbol (default: 'auto')
- **Ventajas**: Eficiente, funciona bien con datos de alta dimensionalidad
- **Desventajas**: Puede tener problemas con datos muy ruidosos

#### Local Outlier Factor (LOF)
- **Tipo**: Basado en densidad local
- **Parámetros**:
  - `n_neighbors`: Número de vecinos (default: 20)
  - `contamination`: Proporción esperada de anomalías (default: 0.1)
  - `metric`: Métrica de distancia (default: 'euclidean')
- **Ventajas**: Detecta anomalías locales, no requiere distribución específica
- **Desventajas**: Computacionalmente costoso para datasets grandes

#### One-Class SVM
- **Tipo**: Basado en Support Vector Machines
- **Parámetros**:
  - `nu`: Proporción esperada de anomalías (default: 0.1)
  - `kernel`: Tipo de kernel ('rbf', 'linear', 'poly', 'sigmoid')
  - `gamma`: Coeficiente del kernel (default: 'scale')
- **Ventajas**: Flexible con diferentes kernels, bueno para datos no lineales
- **Desventajas**: Puede ser lento con muchos datos

#### Autoencoders
- **Tipo**: Red neuronal para reconstrucción
- **Parámetros**:
  - `encoding_dim`: Dimensión de la capa de codificación (default: 32)
  - `hidden_layers`: Capas ocultas (default: [64, 32])
  - `epochs`: Épocas de entrenamiento (default: 50)
  - `threshold_percentile`: Percentil para umbral (default: 95)
- **Ventajas**: Captura relaciones no lineales complejas
- **Desventajas**: Requiere TensorFlow, más lento de entrenar

### 4.3. Interpretación

Las anomalías detectadas pueden indicar:
- Batallas con combinaciones inusuales de cartas
- Jugadores con estrategias atípicas
- Errores en los datos
- Patrones emergentes o novedosos

## 5. Reglas de Asociación

### 5.1. Objetivo

Descubrir relaciones frecuentes entre cartas, identificando qué cartas tienden a aparecer juntas en los mazos.

### 5.2. Algoritmos Implementados

#### Apriori Algorithm
- **Tipo**: Algoritmo clásico de minería de reglas
- **Parámetros**:
  - `min_support`: Soporte mínimo (default: 0.1)
  - `min_confidence`: Confianza mínima (default: 0.5)
  - `min_lift`: Lift mínimo (default: 1.0)
  - `max_len`: Longitud máxima de itemsets
- **Ventajas**: Bien establecido, interpretable
- **Desventajas**: Puede ser lento con muchos items

#### FP-Growth
- **Tipo**: Algoritmo basado en árboles de patrones frecuentes
- **Parámetros**: Mismos que Apriori
- **Ventajas**: Generalmente más rápido que Apriori, especialmente con datasets grandes
- **Desventajas**: Puede usar más memoria

### 5.3. Métricas

- **Support**: Frecuencia de aparición del itemset (A y B juntos)
  - `support(A → B) = P(A ∪ B)`
- **Confidence**: Probabilidad de B dado A
  - `confidence(A → B) = P(B|A) = support(A ∪ B) / support(A)`
- **Lift**: Mide la fuerza de la asociación
  - `lift(A → B) = confidence(A → B) / support(B)`
  - Lift > 1: Asociación positiva
  - Lift = 1: Sin asociación
  - Lift < 1: Asociación negativa

### 5.4. Interpretación

Las reglas de asociación revelan:
- Combinaciones de cartas frecuentes
- Sinergias entre cartas
- Estrategias de construcción de mazos
- Cartas que rara vez aparecen juntas

## 6. Flujo de Trabajo

### 6.1. Ejecución

```bash
# Clustering
kedro run --pipeline=unsupervised_learning

# Reducción de dimensionalidad
kedro run --pipeline=dimensionality_reduction

# Detección de anomalías
kedro run --pipeline=anomaly_detection

# Reglas de asociación
kedro run --pipeline=association_rules
```

### 6.2. Análisis de Resultados

Los resultados se guardan en:
- **Modelos**: `data/06_models/`
- **Métricas**: `data/07_model_output/`
- **Comparaciones**: `data/08_reporting/`

### 6.3. Visualización

Usar el notebook `05_unsupervised_learning.ipynb` para:
- Visualizar métricas de clustering
- Analizar varianza explicada de PCA
- Comparar métodos de detección de anomalías
- Explorar reglas de asociación

## 7. Mejores Prácticas

1. **Normalización**: Siempre normalizar datos antes de clustering y reducción de dimensionalidad
2. **Selección de parámetros**: Usar validación cruzada o métodos como Elbow para clustering
3. **Interpretación**: Combinar múltiples métricas para evaluar modelos
4. **Visualización**: Usar reducción de dimensionalidad para explorar datos
5. **Anomalías**: Validar anomalías detectadas con conocimiento del dominio

## 8. Limitaciones y Consideraciones

- **Escalabilidad**: Algunos algoritmos (Hierarchical, LOF) pueden ser lentos con muchos datos
- **Interpretabilidad**: UMAP y Autoencoders son menos interpretables que PCA
- **Parámetros**: Requieren ajuste según el dataset específico
- **Datos**: La calidad de los resultados depende de la calidad de los datos de entrada

## 9. Referencias

- Scikit-learn Documentation: https://scikit-learn.org/
- UMAP Documentation: https://umap-learn.readthedocs.io/
- MLxtend Documentation: https://rasbt.github.io/mlxtend/
- PyOD Documentation: https://pyod.readthedocs.io/

