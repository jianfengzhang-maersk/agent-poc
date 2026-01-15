# agent_poc/nodes/planning/adapters.py

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Iterable, Type

from pydantic import BaseModel

from agent_poc.semantic_layer.engine import ToolInfo
from agent_poc.semantic_layer.ontology import EntitySchema, RelationSchema
from agent_poc.semantic_layer.generated_models.container import Container
from agent_poc.semantic_layer.generated_models.facility import Facility
from agent_poc.semantic_layer.generated_models.containerevent import Containerevent


def tools_to_dict(tools: Iterable[ToolInfo]) -> List[Dict[str, Any]]:
    """Convert ToolInfo dataclasses to JSON-serializable dicts for the planner."""
    result: List[Dict[str, Any]] = []

    for t in tools:
        # ToolInfo is already a clean structure, use asdict directly
        payload = asdict(t)
        # Remove handler if it can't be serialized (planner only needs schema, not execution)
        payload.pop("handler", None)
        result.append(payload)

    return result


def entities_to_dict(entities: Iterable[EntitySchema]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in entities:
        out.append(
            {
                "name": e.name,
                "description": e.description,
                "synonyms": list(e.synonyms),
                "relationships": {
                    rel_name: {
                        "target": rel_spec.target,
                        "description": rel_spec.description,
                    }
                    for rel_name, rel_spec in e.relationships.items()
                },
            }
        )
    return out


def relations_to_dict(relations: Iterable[RelationSchema]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in relations:
        out.append(
            {
                "name": r.name,
                "from_entity": r.from_entity,
                "to_entity": r.to_entity,
                "description": r.description,
            }
        )
    return out


def normalize_model_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove 'format: date-time' to prevent LLM from outputting ISO8601 timestamps.
    Keep the fact that the field is a string, but not the date-time format.
    """
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            if prop_schema.get("format") == "date-time":
                prop_schema.pop("format", None)
    return schema


def models_to_dict(models: Dict[str, Type[BaseModel]]) -> Dict[str, Dict[str, Any]]:
    """
    Convert Pydantic models into a compact json schema dict for the planner.

    models: mapping from entity_name → Pydantic class
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name, model_cls in models.items():
        schema = model_cls.model_json_schema()
        # Can simplify/rename fields if needed, for now pass as-is to planner
        out[name] = schema

    model_schemas = {k: normalize_model_schema(v) for k, v in out.items()}
    return model_schemas


if __name__ == "__main__":
    import json
    from agent_poc.semantic_layer.engine import semantic_layer

    print("=" * 80)
    print("Adapters Module Demo")
    print("=" * 80)

    # ============================================================
    # Example 1: tools_to_dict()
    # ============================================================
    print("\n[Example 1] tools_to_dict() - Convert ToolInfo to LLM-readable dict")
    print("-" * 80)

    # Get some tools
    city_facility_tools = semantic_layer.get_tools_for_relation(
        "City", "has_facility", "Facility"
    )

    if city_facility_tools:
        print("\nOriginal ToolInfo object (first 1):")
        tool = city_facility_tools[0]
        print(f"  name: {tool.name}")
        print(f"  type: {type(tool)}")
        print(f"  handler: {tool.handler} (function object, not serializable)")
        print(f"  input_schema: {tool.input_schema}")

        # Convert
        converted = tools_to_dict(city_facility_tools)
        print("\nConverted dict:")
        print(json.dumps(converted[0], indent=2, ensure_ascii=False))
        print("\n✓ 'handler' field removed, safe to pass to LLM")

    # ============================================================
    # Example 2: entities_to_dict()
    # ============================================================
    print("\n" + "=" * 80)
    print("[Example 2] entities_to_dict() - Convert EntitySchema to simplified dict")
    print("-" * 80)

    city_entity = semantic_layer.get_entity("City")
    container_entity = semantic_layer.get_entity("Container")

    if city_entity and container_entity:
        print("\nOriginal EntitySchema object:")
        print(f"  City.name: {city_entity.name}")
        print(f"  City.description: {city_entity.description}")
        print(f"  City.relationships: {list(city_entity.relationships.keys())}")

        # Convert
        converted = entities_to_dict([city_entity, container_entity])
        print("\nConverted dict (City):")
        print(json.dumps(converted[0], indent=2, ensure_ascii=False))
        print("\n✓ Contains name, description, synonyms, relationships, etc.")

    # ============================================================
    # Example 3: relations_to_dict()
    # ============================================================
    print("\n" + "=" * 80)
    print("[Example 3] relations_to_dict() - Convert RelationSchema to dict")
    print("-" * 80)

    # Get some relations
    city_relations = semantic_layer.list_relations_from("City")

    if city_relations:
        print("\nOriginal RelationSchema objects (first 2):")
        for rel in city_relations[:2]:
            print(f"  {rel.from_entity} --{rel.name}--> {rel.to_entity}")
            print(f"    description: {rel.description}")

        # Convert
        converted = relations_to_dict(city_relations[:2])
        print("\nConverted dict:")
        print(json.dumps(converted, indent=2, ensure_ascii=False))
        print("\n✓ Flattened relation representation, easy for LLM to understand")

    # ============================================================
    # Example 4: models_to_dict()
    # ============================================================
    print("\n" + "=" * 80)
    print("[Example 4] models_to_dict() - Convert Pydantic models to JSON Schema")
    print("-" * 80)

    models = {
        "Container": Container,
        "Facility": Facility,
    }

    print("\nOriginal Pydantic model classes:")
    print(f"  Container: {Container}")
    print(f"  Facility: {Facility}")

    # Convert
    converted = models_to_dict(models)

    print("\nConverted JSON Schema (Container - partial fields):")
    container_schema = converted["Container"]
    print(f"  title: {container_schema.get('title')}")
    print(f"  type: {container_schema.get('type')}")
    print(f"  properties count: {len(container_schema.get('properties', {}))}")

    # Show some fields
    if "properties" in container_schema:
        print("\n  Sample fields:")
        for field_name in list(container_schema["properties"].keys())[:3]:
            field_info = container_schema["properties"][field_name]
            print(f"    - {field_name}: {field_info.get('type', 'N/A')}")
            if "description" in field_info:
                print(f"      description: {field_info['description']}")

    print("\n✓ Pydantic models converted to standard JSON Schema format")

    # ============================================================
    # Example 5: normalize_model_schema() - date-time format cleanup
    # ============================================================
    print("\n" + "=" * 80)
    print("[Example 5] normalize_model_schema() - Remove date-time format")
    print("-" * 80)

    # Create a test schema with date-time
    test_schema = {
        "properties": {
            "created_at": {
                "type": "string",
                "format": "date-time",
                "description": "Creation time",
            },
            "name": {"type": "string", "description": "Name"},
        }
    }

    print("\nBefore normalization:")
    print(json.dumps(test_schema, indent=2, ensure_ascii=False))

    normalized = normalize_model_schema(test_schema)

    print("\nAfter normalization:")
    print(json.dumps(normalized, indent=2, ensure_ascii=False))
    print(
        "\n✓ 'format: date-time' removed to prevent LLM from generating ISO8601 format"
    )

    # ============================================================
    # Comprehensive Example: Preparing complete inputs for Planner
    # ============================================================
    print("\n" + "=" * 80)
    print("[Comprehensive Example] Preparing complete inputs for TypeAwarePlanner")
    print("-" * 80)

    # Scenario: Query "How many containers were gated out of Sydney terminal?"
    print(
        "\nScenario: Preparing to answer 'How many containers were gated out of Sydney terminal?'"
    )

    # 1. Prepare tools
    relevant_tools = semantic_layer.get_tools_for_relation(
        "City", "has_facility", "Facility"
    ) + semantic_layer.get_tools_for_relation(
        "Facility", "hosts_event", "ContainerEvent"
    )
    candidate_tools = tools_to_dict(relevant_tools)
    print(f"\n1. Candidate tools: {len(candidate_tools)} tools")
    for tool in candidate_tools:
        print(f"   - {tool['name']}")

    # 2. Prepare entity schemas
    relevant_entities = [
        semantic_layer.get_entity("City"),
        semantic_layer.get_entity("Facility"),
        semantic_layer.get_entity("ContainerEvent"),
    ]
    entity_schemas = entities_to_dict([e for e in relevant_entities if e])
    print(f"\n2. Entity schemas: {len(entity_schemas)} entities")
    for entity in entity_schemas:
        print(f"   - {entity['name']}: {entity['description'][:50]}...")

    # 3. Prepare relation schemas
    relevant_relations = [
        semantic_layer.get_relation("City", "has_facility", "Facility"),
        semantic_layer.get_relation("Facility", "hosts_event", "ContainerEvent"),
    ]
    relation_schemas = relations_to_dict([r for r in relevant_relations if r])
    print(f"\n3. Relation schemas: {len(relation_schemas)} relations")
    for rel in relation_schemas:
        print(f"   - {rel['from_entity']} --{rel['name']}--> {rel['to_entity']}")

    # 4. Prepare Pydantic model schemas
    pydantic_models = {
        "Facility": Facility,
        "ContainerEvent": Containerevent,
        "Container": Container,
    }
    model_schemas = models_to_dict(pydantic_models)
    print(f"\n4. Pydantic model schemas: {len(model_schemas)} models")
    for model_name in model_schemas.keys():
        print(f"   - {model_name}")

    print("\n✓ All inputs prepared and ready to pass to TypeAwarePlanner!")
    print("\nNote: All converted data is JSON-serializable and")
    print("      can be safely passed to LLM via DSPy for inference.")
    print("\n" + "=" * 80)
