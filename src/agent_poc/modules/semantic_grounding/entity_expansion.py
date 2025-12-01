# semantic_grounding/entity_expansion.py

from typing import Dict, List, Set, Tuple

from agent_poc.semantic_layer.ontology import RelationKey


def expand_entities(
    step1_entities: List[str]
) -> List[str]:
    """
    Step 2.3: Deterministic entity expansion.

    Input:
      - step1_entities: ["City", ...]
    Output:
      - expanded_entity_types: list of entity type strings
    """

    # 1. collect initial entity types
    entity_types: Set[str] = {e["type"] for e in step1_entities}

    # 2. filter relations = yes
    active_relations: List[RelationKey] = [
        rel for rel, val in relevant_relations.items() if str(val).lower() == "yes"
    ]

    # 3. one-pass deterministic expansion
    changed = True
    while changed:
        changed = False

        for rel in active_relations:
            source, _, target = rel
            # If source is included → include target
            if source in entity_types and target not in entity_types:
                entity_types.add(target)
                changed = True

            # If target is included → include source
            if target in entity_types and source not in entity_types:
                entity_types.add(source)
                changed = True

    return sorted(entity_types), active_relations


if __name__ == "__main__":
    sample_entities = [
        {"type": "City", "value": "Sydney"},
        {"type": "Container", "value": "HASU1234567"},
    ]

    sample_relations: Dict[RelationKey, str] = {
        ("City", "has_facility", "Facility"): "yes",
        ("Facility", "occurs_at", "ContainerEvent"): "yes",
        ("ContainerEvent", "for_container", "Container"): "yes",
        ("Shipment", "has_container", "Container"): "no",
    }

    expanded_entities, used_relations = expand_entities(
        sample_entities, sample_relations
    )

    print("Expanded entity types:", expanded_entities)
    print("Active relations:")
    for rel in used_relations:
        print(f"  {rel[0]} -- {rel[1]} --> {rel[2]}")
