"""Pipelines del proyecto ML Clash Royale."""

from .business_understanding import create_business_understanding_pipeline
from .eda import create_eda_pipeline
from .data_preparation import create_data_preparation_pipeline
from .feature_engineering import create_feature_engineering_pipeline
from .classification import create_classification_pipeline
from .regression import create_regression_pipeline
from .unsupervised_learning import create_unsupervised_learning_pipeline
from .dimensionality_reduction import create_dimensionality_reduction_pipeline
from .anomaly_detection import create_anomaly_detection_pipeline
from .association_rules import create_association_rules_pipeline
from .reporting import create_reporting_pipeline

__all__ = [
    "create_business_understanding_pipeline",
    "create_eda_pipeline",
    "create_data_preparation_pipeline",
    "create_feature_engineering_pipeline",
    "create_classification_pipeline",
    "create_regression_pipeline",
    "create_unsupervised_learning_pipeline",
    "create_dimensionality_reduction_pipeline",
    "create_anomaly_detection_pipeline",
    "create_association_rules_pipeline",
    "create_reporting_pipeline"
]

