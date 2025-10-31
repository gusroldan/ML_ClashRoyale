"""Pipeline para feature engineering."""

from kedro.pipeline import Pipeline, node
from typing import Dict
from proyecto_ml_clashroyale.pipelines.nodes import feature_engineering_nodes


def create_feature_engineering_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de feature engineering.
    
    Returns:
        Pipeline de feature engineering
    """
    
    return Pipeline(
        [
            node(
                func=feature_engineering_nodes.create_card_dictionary,
                inputs="card_master_list",
                outputs="card_dictionary",
                name="create_card_dictionary_node",
                tags=["feature_engineering", "cards"],
            ),
            node(
                func=feature_engineering_nodes.create_balanced_dataset,
                inputs=["combined_dataset", "params:feature_engineering"],
                outputs="balanced_dataset",
                name="create_balanced_dataset_node",
                tags=["feature_engineering", "balance"],
            ),
            node(
                func=feature_engineering_nodes.encode_cards_multihot,
                inputs=["balanced_dataset", "card_dictionary"],
                outputs="features_multihot",
                name="encode_cards_multihot_node",
                tags=["feature_engineering", "encoding"],
            ),
            node(
                func=feature_engineering_nodes.create_aggregate_features,
                inputs=["balanced_dataset", "features_multihot"],
                outputs="features_aggregate",
                name="create_aggregate_features_node",
                tags=["feature_engineering", "aggregate"],
            ),
            node(
                func=feature_engineering_nodes.combine_features,
                inputs=["features_multihot", "features_aggregate"],
                outputs="features_combined",
                name="combine_features_node",
                tags=["feature_engineering", "merge"],
            ),
            node(
                func=feature_engineering_nodes.split_train_test,
                inputs=["features_combined", "params:feature_engineering"],
                outputs=["train_data", "test_data"],
                name="split_train_test_node",
                tags=["feature_engineering", "split"],
            ),
        ]
    )

