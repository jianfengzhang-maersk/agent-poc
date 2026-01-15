"""Semantic grounding pipeline that composes relation filtering and entity expansion."""

from __future__ import annotations

from typing import List, Tuple

from agent_poc.modules.semantic_grounding.entity_expansion import expand_entities
from agent_poc.modules.semantic_grounding.relation_discovery import discover_relations
from agent_poc.modules.semantic_grounding.relation_filtering import RelationFiltering
from agent_poc.semantic_layer.ontology import RelationKey
from agent_poc.semantic_layer.engine import semantic_layer


def run_semantic_grounding(
    query: str,
    entities: List[dict],
    intent: str,
    filtering_model: RelationFiltering | None = None,
) -> Tuple[List[str], List[RelationKey]]:
    """Pipeline Step 2: relation filtering + entity expansion."""

    if not entities:
        return [], []

    filtering_model = filtering_model or RelationFiltering()

    discovered_relations = discover_relations(entities)
    filtered_relations = filtering_model(
        query=query,
        intent=intent,
        relations=discovered_relations,
    )

    expanded_entities, active_relations = expand_entities(
        step1_entities=entities,
        relevant_relations=filtered_relations,
    )

    return expanded_entities, active_relations


if __name__ == "__main__":
    from agent_poc.utils.dspy_helper import DspyHelper
    from agent_poc.modules.query_understanding.query_understanding import (
        QueryUnderstanding,
    )
    from agent_poc.semantic_layer.engine import semantic_layer, ontology_entities

    DspyHelper.init_kimi()

    qu = QueryUnderstanding(ontology_entities)
    qu.load(
        "src/agent_poc/modules/query_understanding/query_understanding_optimized_2.json"
    )

    sample_query = (
        "How many containers were gated out of Sydney terminal on 20 July 2025?"
    )
    qu_result = qu(query=sample_query)

    filtering_model = RelationFiltering(batch_size=4)
    expanded_entities, active_relations = run_semantic_grounding(
        query=sample_query,
        entities=qu_result.entities,
        intent=qu_result.intent,
        filtering_model=filtering_model,
    )

    expanded_entities = [
        entity for entity in expanded_entities if semantic_layer.has_entity(entity)
    ]
    print("Expanded entities:", expanded_entities)
    print("Relevant relations:")
    for rel in active_relations:
        print(f"  {rel[0]} --{rel[1]}--> {rel[2]}")
