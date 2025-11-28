"""Pipeline para reglas de asociación."""
## Objetivo: encontrar patrones de asociación entre items (cartas) en las transacciones (batallas).
from kedro.pipeline import Pipeline, node
from typing import Dict
from proyecto_ml_clashroyale.pipelines.nodes import association_rules_nodes


def create_association_rules_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de reglas de asociación.
    
    Returns:
        Pipeline de reglas de asociación
    """
    
    return Pipeline(
        [
            node(
                func=association_rules_nodes.find_association_rules_apriori,
                inputs=["train_data", "params:association_rules"],
                outputs=["apriori_result", "apriori_metrics"],
                name="find_association_rules_apriori_node",
                tags=["association_rules", "apriori"],
            ),
            node(
                func=association_rules_nodes.find_association_rules_fpgrowth,
                inputs=["train_data", "params:association_rules"],
                outputs=["fpgrowth_result", "fpgrowth_metrics"],
                name="find_association_rules_fpgrowth_node",
                tags=["association_rules", "fpgrowth"],
            ),
            node(
                func=association_rules_nodes.create_association_rules_comparison,
                inputs=["apriori_result", "fpgrowth_result"],
                outputs="association_rules_comparison",
                name="create_association_rules_comparison_node",
                tags=["association_rules", "evaluation"],
            ),
        ]
    )

