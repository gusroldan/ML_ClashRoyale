"""Pipeline para análisis exploratorio de datos (EDA)."""

from kedro.pipeline import Pipeline, node

from .nodes import eda_nodes


def create_eda_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de análisis exploratorio de datos.
    
    Returns:
        Pipeline de EDA configurado
    """
    return Pipeline(
        [
            node(
                func=eda_nodes.analyze_rarity_distributions,
                inputs=[
                    "Combates1_cleaned",
                    "Combates2_cleaned", 
                    "Combates3_cleaned"
                ],
                outputs="rarity_distributions_analysis",
                name="analyze_rarity_distributions",
                tags=["eda", "rarity"]
            ),
            node(
                func=eda_nodes.analyze_most_used_cards,
                inputs=[
                    "Combates1_cleaned",
                    "Combates2_cleaned",
                    "Combates3_cleaned",
                    "card_master_list"
                ],
                outputs="most_used_cards_analysis",
                name="analyze_most_used_cards",
                tags=["eda", "cards"]
            ),
            node(
                func=eda_nodes.analyze_win_conditions_usage,
                inputs=[
                    "Combates1_cleaned",
                    "Combates2_cleaned",
                    "Combates3_cleaned",
                    "wincons",
                    "card_master_list"
                ],
                outputs="win_conditions_usage_analysis",
                name="analyze_win_conditions_usage",
                tags=["eda", "win_conditions"]
            ),
            node(
                func=eda_nodes.generate_eda_summary,
                inputs=[
                    "rarity_distributions_analysis",
                    "most_used_cards_analysis",
                    "win_conditions_usage_analysis"
                ],
                outputs="eda_summary",
                name="generate_eda_summary",
                tags=["eda", "summary"]
            )
        ]
    )

