"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines import create_business_understanding_pipeline, create_eda_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    
    # Pipeline específico para comprensión del negocio
    pipelines["business_understanding"] = create_business_understanding_pipeline()
    
    # Pipeline específico para análisis exploratorio de datos
    pipelines["eda"] = create_eda_pipeline()
    
    # Pipeline por defecto incluye todos los pipelines
    pipelines["__default__"] = sum(pipelines.values())
    
    return pipelines
