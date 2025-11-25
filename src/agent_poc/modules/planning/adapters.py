# agent_poc/nodes/planning/adapters.py

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Iterable

from agent_poc.semantic_layer.engine import ToolInfo
from agent_poc.semantic_layer.ontology import EntitySchema, RelationSchema
from agent_poc.semantic_layer.generated_models.container import Container
from agent_poc.semantic_layer.generated_models.shipment import Shipment
from agent_poc.semantic_layer.generated_models.facility import Facility
from agent_poc.semantic_layer.generated_models.containerevent import Containerevent


def tools_to_candidate_tools(tools: Iterable[ToolInfo]) -> List[Dict[str, Any]]:
    """Convert ToolInfo dataclasses to JSON-serializable dicts for the planner."""
    result: List[Dict[str, Any]] = []

    for t in tools:
        # ToolInfo 里已经是很干净的结构，用 asdict 即可
        payload = asdict(t)
        # 如果 handler 不能序列化，可以删掉它（planner 不需要执行，只需要 schema）
        payload.pop("handler", None)
        result.append(payload)

    return result


def entities_to_schemas(entities: Iterable[EntitySchema]) -> List[Dict[str, Any]]:
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


def relations_to_schemas(relations: Iterable[RelationSchema]) -> List[Dict[str, Any]]:
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


from typing import Type, Dict
from pydantic import BaseModel


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

def models_to_schemas(models: Dict[str, Type[BaseModel]]) -> Dict[str, Dict[str, Any]]:
    """
    Convert Pydantic models into a compact json schema dict for the planner.

    models: mapping from entity_name → Pydantic class
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name, model_cls in models.items():
        schema = model_cls.model_json_schema()
        # 这里可以按需要精简/重命名字段，先直接原样给 planner
        out[name] = schema
        
    model_schemas = {k: normalize_model_schema(v) for k, v in out.items()}
    return model_schemas
