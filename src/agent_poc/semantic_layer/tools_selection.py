# semantic_layer/tool_selection.py

from typing import Dict, List
from agent_poc.semantic_layer.tools_registry import TOOLS_REGISTRY
from agent_poc.semantic_layer.ontology import RelationKey

from agent_poc.semantic_layer.engine import semantic_layer


def select_tools(
    expanded_entities: List[str],
    active_relations: List[RelationKey],
) -> List[str]:
    """
    Step 3.1: Deterministic tool candidate filtering.

    Input:
      expanded_entities: list of entity types returned by Step 2.3
      active_relations: list of relevant Relation dataclass objects from Step 2.2

    Output:
      List[Dict] – the tool metadata selected from TOOLS_REGISTRY
    """

    selected_tools = []

    for entity in expanded_entities:
        selected_tools.extend(semantic_layer.get_tools_for_entity(entity))

    for relation in active_relations:
        selected_tools.extend(
            semantic_layer.get_tools_for_relation(relation[0], relation[1], relation[2])
        )
    return selected_tools


if __name__ == "__main__":

    from agent_poc.semantic_layer.engine import semantic_layer, ontology_entities

    question = "How many containers were gated out of Sydney terminal on 20 July 2025?"
    print("Question:", question)

    expanded_entities = ["City", "Facility", "ContainerEvent", "Container"]

    active_relations = [
        ("City", "has_facility", "Facility"),
        ("Facility", "hosts_event", "ContainerEvent"),
        ("Container", "has_event", "ContainerEvent"),
    ]

    selected = select_tools(expanded_entities, active_relations)
    selected_tool_names = [tool.name for tool in selected]
    print("Selected tools:", selected_tool_names)
