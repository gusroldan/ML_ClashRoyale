"""Nodos para la comprensión del negocio (Fase 1 CRISP-DM)."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


def clean_and_rename_first_column(df: pd.DataFrame) -> pd.DataFrame:
    """Limpiar y renombrar la primera columna del dataset.
    
    Args:
        df: DataFrame con la primera columna sin nombre
        
    Returns:
        DataFrame con la primera columna renombrada como 'battle_id'
    """
    logger.info("Limpiando y renombrando primera columna...")
    
    # Crear una copia del DataFrame
    df_cleaned = df.copy()
    
    # Renombrar la primera columna (que no tiene nombre)
    first_column = df_cleaned.columns[0]
    df_cleaned = df_cleaned.rename(columns={first_column: 'battle_id'})
    
    # Verificar que la primera columna ahora se llama 'battle_id'
    logger.info(f"Primera columna renombrada a: {df_cleaned.columns[0]}")
    logger.info(f"Shape del dataset: {df_cleaned.shape}")
    
    return df_cleaned


def analyze_business_objectives(combates1: pd.DataFrame, combates2: pd.DataFrame, 
                              combates3: pd.DataFrame, card_master_list: pd.DataFrame, 
                              wincons: pd.DataFrame) -> Dict[str, Any]:
    """Analizar los objetivos del negocio basado en los datos disponibles.
    
    Args:
        combates1: Dataset de combates 1
        combates2: Dataset de combates 2  
        combates3: Dataset de combates 3
        card_master_list: Lista maestra de cartas
        wincons: Lista de win conditions
        
    Returns:
        Diccionario con análisis de objetivos del negocio
    """
    logger.info("Analizando objetivos del negocio...")
    
    # Combinar todos los datasets de combates para análisis inicial
    all_combates = pd.concat([combates1, combates2, combates3], ignore_index=True)
    
    # Análisis básico de los datos disponibles
    total_battles = len(all_combates)
    total_cards = len(card_master_list)
    total_wincons = len(wincons)
    
    # Objetivos principales identificados
    business_objectives = {
        "proyecto": "Análisis de Estrategias en Clash Royale",
        "objetivos_principales": [
            "Identificar las cartas más utilizadas en los mazos",
            "Analizar las win conditions más efectivas",
            "Estudiar la distribución de rarezas en los mazos",
            "Comprender patrones de éxito en las batallas"
        ],
        "datos_disponibles": {
            "total_batallas": total_battles,
            "total_cartas": total_cards,
            "total_wincons": total_wincons,
            "periodo_temporal": "Diciembre 2020 - Enero 2021"
        },
        "preguntas_negocio": [
            "¿Qué cartas son más populares en los mazos ganadores?",
            "¿Cuáles son las win conditions más efectivas?",
            "¿Cómo influye la rareza de las cartas en el éxito?",
            "¿Existen patrones específicos en los mazos exitosos?"
        ],
        "valor_esperado": "Mejorar la comprensión de estrategias efectivas para jugadores y desarrolladores"
    }
    
    logger.info(f"Análisis completado: {total_battles} batallas, {total_cards} cartas")
    return business_objectives


def evaluate_current_situation(combates1: pd.DataFrame, combates2: pd.DataFrame, 
                             combates3: pd.DataFrame) -> Dict[str, Any]:
    """Evaluar la situación actual de los datos.
    
    Args:
        combates1: Dataset de combates 1
        combates2: Dataset de combates 2
        combates3: Dataset de combates 3
        
    Returns:
        Diccionario con evaluación de la situación actual
    """
    logger.info("Evaluando situación actual de los datos...")
    
    # Combinar datasets
    all_combates = pd.concat([combates1, combates2, combates3], ignore_index=True)
    
    # Análisis de calidad de datos
    missing_values = all_combates.isnull().sum()
    data_types = all_combates.dtypes.to_dict()
    
    # Información básica de los datasets
    situation_evaluation = {
        "datasets_disponibles": {
            "combates_12272020": {
                "registros": len(combates1),
                "columnas": len(combates1.columns)
            },
            "combates_12312020": {
                "registros": len(combates2), 
                "columnas": len(combates2.columns)
            },
            "combates_01042021": {
                "registros": len(combates3),
                "columnas": len(combates3.columns)
            }
        },
        "calidad_datos": {
            "total_registros": len(all_combates),
            "total_columnas": len(all_combates.columns),
            "valores_faltantes": missing_values.to_dict(),
            "tipos_datos": {str(k): str(v) for k, v in data_types.items()}
        },
        "fortalezas": [
            "Múltiples períodos temporales disponibles",
            "Datos estructurados y consistentes",
            "Información detallada de cartas y batallas"
        ],
        "limitaciones": [
            "Datos históricos limitados a 3 períodos",
            "Posibles valores faltantes en algunos campos",
            "Necesidad de limpieza y validación"
        ]
    }
    
    logger.info(f"Situación evaluada: {len(all_combates)} registros totales")
    return situation_evaluation


def define_ml_objectives(business_objectives: Dict[str, Any], 
                        current_situation: Dict[str, Any]) -> Dict[str, Any]:
    """Definir objetivos específicos de Machine Learning.
    
    Args:
        business_objectives: Análisis de objetivos del negocio
        current_situation: Evaluación de la situación actual
        
    Returns:
        Diccionario con objetivos de ML definidos
    """
    logger.info("Definiendo objetivos de Machine Learning...")
    
    ml_objectives = {
        "objetivos_ml": [
            {
                "objetivo": "Clasificación de Cartas Populares",
                "descripcion": "Identificar las cartas más utilizadas en mazos exitosos",
                "tipo": "Análisis descriptivo",
                "metricas": ["Frecuencia de uso", "Tasa de éxito"]
            },
            {
                "objetivo": "Análisis de Win Conditions",
                "descripcion": "Determinar las win conditions más efectivas",
                "tipo": "Análisis de efectividad",
                "metricas": ["Tasa de victoria", "Frecuencia de uso"]
            },
            {
                "objetivo": "Distribución de Rarezas",
                "descripcion": "Analizar cómo la rareza afecta el éxito del mazo",
                "tipo": "Análisis estadístico",
                "metricas": ["Distribución por rareza", "Correlación con éxito"]
            }
        ],
        "criterios_exito": [
            "Identificar al menos 10 cartas más populares",
            "Determinar las 5 win conditions más efectivas",
            "Establecer patrones claros en distribución de rarezas"
        ],
        "limitaciones_tecnicas": [
            "Datos históricos limitados",
            "Posible sesgo temporal en los datos",
            "Necesidad de validación cruzada"
        ]
    }
    
    logger.info("Objetivos de ML definidos exitosamente")
    return ml_objectives


def generate_project_plan(business_objectives: Dict[str, Any], 
                         current_situation: Dict[str, Any],
                         ml_objectives: Dict[str, Any]) -> Dict[str, Any]:
    """Generar plan detallado del proyecto.
    
    Args:
        business_objectives: Análisis de objetivos del negocio
        current_situation: Evaluación de la situación actual
        ml_objectives: Objetivos de ML definidos
        
    Returns:
        Diccionario con plan del proyecto
    """
    logger.info("Generando plan del proyecto...")
    
    project_plan = {
        "fases_crisp_dm": {
            "fase_1": {
                "nombre": "Comprensión del Negocio",
                "estado": "En progreso",
                "actividades": [
                    "Definir objetivos del proyecto",
                    "Evaluar situación actual",
                    "Determinar objetivos de ML",
                    "Producir plan del proyecto"
                ]
            },
            "fase_2": {
                "nombre": "Comprensión de los Datos",
                "estado": "Pendiente",
                "actividades": [
                    "Recolectar datos iniciales",
                    "Describir los datos",
                    "Explorar los datos (EDA)",
                    "Verificar calidad de los datos"
                ]
            },
            "fase_3": {
                "nombre": "Preparación de los Datos",
                "estado": "Pendiente",
                "actividades": [
                    "Seleccionar datos relevantes",
                    "Limpiar los datos",
                    "Construir nuevas variables",
                    "Integrar datos de múltiples fuentes"
                ]
            }
        },
        "entregables": [
            "Análisis de objetivos del negocio",
            "Evaluación de situación actual",
            "Objetivos de ML definidos",
            "Plan del proyecto",
            "Resumen ejecutivo"
        ],
        "recursos_necesarios": [
            "Datos de batallas históricas",
            "Catálogo de cartas",
            "Lista de win conditions",
            "Herramientas de análisis (pandas, numpy, matplotlib)"
        ],
        "cronograma": {
            "fase_1": "1-2 días",
            "fase_2": "3-5 días", 
            "fase_3": "2-3 días"
        }
    }
    
    logger.info("Plan del proyecto generado exitosamente")
    return project_plan


def create_business_summary(business_objectives: Dict[str, Any],
                          current_situation: Dict[str, Any],
                          ml_objectives: Dict[str, Any],
                          project_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Crear resumen ejecutivo de la comprensión del negocio.
    
    Args:
        business_objectives: Análisis de objetivos del negocio
        current_situation: Evaluación de la situación actual
        ml_objectives: Objetivos de ML definidos
        project_plan: Plan del proyecto
        
    Returns:
        Diccionario con resumen ejecutivo
    """
    logger.info("Creando resumen ejecutivo...")
    
    summary = {
        "resumen_ejecutivo": {
            "proyecto": business_objectives["proyecto"],
            "objetivo_principal": "Analizar estrategias efectivas en Clash Royale",
            "datos_disponibles": f"{current_situation['calidad_datos']['total_registros']:,} batallas históricas",
            "objetivos_clave": business_objectives["objetivos_principales"],
            "valor_esperado": business_objectives["valor_esperado"]
        },
        "siguientes_pasos": [
            "Proceder con Fase 2: Comprensión de los Datos",
            "Realizar EDA detallado de los datasets",
            "Limpiar y preparar datos para análisis",
            "Implementar análisis de cartas populares"
        ],
        "recomendaciones": [
            "Enfocar análisis en mazos ganadores",
            "Considerar factores temporales en los datos",
            "Validar resultados con múltiples métricas",
            "Documentar hallazgos para futuras iteraciones"
        ]
    }
    
    logger.info("Resumen ejecutivo creado exitosamente")
    return summary
