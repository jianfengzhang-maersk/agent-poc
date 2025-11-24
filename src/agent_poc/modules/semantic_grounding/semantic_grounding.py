"""Semantic grounding pipeline that composes relation relevance and graph expansion."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from agent_poc.modules.semantic_grounding.graph_expansion import expand_graph
from agent_poc.modules.semantic_grounding.relation_relevance import RelationRelevance
from agent_poc.semantic_layer.ontology import RelationKey
from agent_poc.semantic_layer.runtime import SemanticLayer


def _collect_candidate_relations(
    semantic_layer: SemanticLayer,
    seed_entities: Sequence[dict],
) -> List[Tuple[str, str, str, str]]:
    """Gather unique ontology relations touching the provided entity types."""

    seen: set[RelationKey] = set()
    candidates: List[Tuple[str, str, str, str]] = []

    for entity in seed_entities:
        entity_type = entity.get("type")
        if not entity_type:
            continue

        for rel in semantic_layer.list_relations(entity_type):
            if rel.key in seen:
                continue
            seen.add(rel.key)
            candidates.append(
                (
                    rel.from_entity,
                    rel.name,
                    rel.to_entity,
                    rel.description or "",
                )
            )

    return candidates


def run_semantic_grounding(
    query: str,
    entities: List[dict],
    intent: str,
    semantic_layer: SemanticLayer,
    relevance_model: RelationRelevance | None = None,
) -> Tuple[List[str], List[RelationKey]]:
    """Pipeline Step 2: relation relevance + graph expansion."""

    if not entities:
        return [], []

    relevance_model = relevance_model or RelationRelevance()

    candidate_relations = _collect_candidate_relations(semantic_layer, entities)
    relevance_scores = relevance_model(
        query=query,
        intent=intent,
        relations=candidate_relations,
    )

    expanded_entities, active_relations = expand_graph(
        step1_entities=entities,
        relevant_relations=relevance_scores,
    )

    return expanded_entities, active_relations


if __name__ == "__main__":
    from agent_poc.utils.dspy_helper import DspyHelper
    from agent_poc.modules.query_understanding.query_understanding import QueryUnderstanding
    from agent_poc.semantic_layer.runtime import semantic_layer, ontology_entities

    DspyHelper.init_kimi()

    qu = QueryUnderstanding(ontology_entities)
    qu.load("src/agent_poc/modules/query_understanding/query_understanding_optimized_2.json")

    sample_query = "How many containers were gated out of Sydney terminal on 20 July 2025?"
    qu_result = qu(query=sample_query)

    relation_model = RelationRelevance(batch_size=4)
    expanded_entities, active_relations = run_semantic_grounding(
        query=sample_query,
        entities=qu_result.entities,
        intent=qu_result.intent,
        semantic_layer=semantic_layer,
        relevance_model=relation_model,
    )

    expanded_entities = [
        entity for entity in expanded_entities if semantic_layer.has_entity(entity)
    ]
    print("Expanded entities:", expanded_entities)
    print("Relevant relations:")
    for rel in active_relations:
        print(f"  {rel[0]} --{rel[1]}--> {rel[2]}")
