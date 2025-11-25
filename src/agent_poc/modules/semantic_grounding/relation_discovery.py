from typing import List, Sequence, Tuple

from agent_poc.semantic_layer.engine import SemanticLayer
from agent_poc.semantic_layer.ontology import RelationKey
from agent_poc.semantic_layer.engine import semantic_layer


def discover_relations(
    seed_entities: Sequence[dict]
) -> List[Tuple[str, str, str, str]]:
    """Discover unique ontology relations touching the provided entity types.
    
    Args:
        seed_entities: List of entity dicts with 'type' field
        semantic_layer: Optional SemanticLayer instance. If None, uses the global default.
    
    Returns:
        List of tuples (from_entity, relation_name, to_entity, description)
    """    
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


if __name__ == "__main__":
    sample_seed_entities = [
        {"type": "City"},
        {"type": "Facility"},
    ]

    relations = discover_relations(sample_seed_entities)
    print("Seed entities:", sample_seed_entities)
    print("Discovered relations:")
    for source, name, target, description in relations:
        print(f"  {source}.{name}->{target} -> {description}")
