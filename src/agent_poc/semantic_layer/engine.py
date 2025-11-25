from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Sequence
from collections import defaultdict

from agent_poc.semantic_layer.ontology import (
    RelationKey,
    EntitySchema,
    RelationSchema,
    load_ontology,
)
from agent_poc.semantic_layer.tools_registry import TOOLS_REGISTRY


DEFAULT_TOOL_MODULES: Sequence[str] = ("agent_poc.semantic_layer.tools",)
ONTOLOGY_SOURCE_PATH = Path(__file__).with_name("ontology_data")


@dataclass
class ToolInfo:
    name: str
    description: str
    input_schema: List[Dict[str, Any]]
    output_type: str
    handler: Callable[..., Any]
    kind: str  # "relation" or "entity"
    associated_relation: Optional[RelationKey] = None
    associated_entity: Optional[str] = None


# -------------------------
# Semantic Layer Engine
# -------------------------


@dataclass
class SemanticLayer:
    # ontology
    entities: Dict[str, EntitySchema]
    relations: Dict[RelationKey, RelationSchema]

    # tools
    tools: Dict[str, ToolInfo]
    tools_by_relation: DefaultDict[RelationKey, List[ToolInfo]]
    tools_by_entity: DefaultDict[str, List[ToolInfo]]

    # ---------- Query helpers ----------

    def get_entity(self, name: str) -> Optional[EntitySchema]:
        """Exact match on entity name (case sensitive)."""
        return self.entities.get(name)

    def has_entity(self, name: str) -> bool:
        """Check if an entity exists in the ontology."""
        return name in self.entities

    def find_entity_by_label(self, label: str) -> Optional[EntitySchema]:
        """
        Fuzzy-ish lookup: match against name or synonyms (case-insensitive).
        This is a simple match implementation. You can replace it with embedding search later.
        """
        label_lower = label.lower()
        for ent in self.entities.values():
            if ent.name.lower() == label_lower:
                return ent
            if any(s.lower() == label_lower for s in ent.synonyms):
                return ent
        return None

    def get_relation(
        self, from_entity: str, name: str, to_entity: str
    ) -> Optional[RelationSchema]:
        return self.relations.get((from_entity, name, to_entity))

    def get_tools_for_relation(
        self, from_entity: str, name: str, to_entity: str
    ) -> List[ToolInfo]:
        return self.tools_by_relation.get((from_entity, name, to_entity), [])

    def get_tools_for_entity(self, entity_name: str) -> List[ToolInfo]:
        return self.tools_by_entity.get(entity_name, [])

    def list_relations_from(self, entity_name: str) -> List[RelationSchema]:
        """List all outgoing relations from an entity."""
        return [r for r in self.relations.values() if r.from_entity == entity_name]

    def list_relations_to(self, entity_name: str) -> List[RelationSchema]:
        """List all incoming relations to an entity."""
        return [r for r in self.relations.values() if r.to_entity == entity_name]

    def list_relations(self, entity_name: str) -> List[RelationSchema]:
        return self.list_relations_from(entity_name) + self.list_relations_to(
            entity_name
        )

    def list_entities(self) -> List[EntitySchema]:
        """List all entities in the ontology."""
        return list(self.entities.values())


def load_tools(tool_modules: Optional[Sequence[str]] = None) -> Dict[str, ToolInfo]:
    modules = tool_modules or DEFAULT_TOOL_MODULES

    # Clear registry before importing modules to avoid stale entries when reloading
    TOOLS_REGISTRY.clear()

    for module_path in modules:
        import_module(module_path)

    tools: Dict[str, ToolInfo] = {}
    for name, meta in TOOLS_REGISTRY.items():
        relation = meta.get("relation")
        entity = meta.get("entity")
        description = meta.get("description", "")
        input_schema = meta.get("input_schema", [])
        output_type = meta.get("output_type", "Any")
        handler = meta.get("fn")

        if relation:
            kind = "relation"
        elif entity:
            kind = "entity"
        else:
            raise ValueError(
                f"Tool '{name}' must declare either an entity or relation association"
            )

        tools[name] = ToolInfo(
            name=name,
            description=description,
            input_schema=list(input_schema),
            output_type=output_type,
            handler=handler,
            kind=kind,
            associated_relation=relation,
            associated_entity=entity,
        )

    return tools


def build_semantic_layer(
    ontology_path: str | Path,
    tool_modules: Optional[Sequence[str]] = None,
) -> SemanticLayer:
    # 1) Load ontology
    entities, relations = load_ontology(ontology_path)

    # 2) Discover tools from decorated functions
    tools = load_tools(tool_modules)

    # 3) Link tools ↔ relations / entities
    tools_by_relation: DefaultDict[RelationKey, List[ToolInfo]] = defaultdict(list)
    tools_by_entity: DefaultDict[str, List[ToolInfo]] = defaultdict(list)

    for tool in tools.values():
        if tool.kind == "relation" and tool.associated_relation is not None:
            tools_by_relation[tool.associated_relation].append(tool)
        elif tool.kind == "entity" and tool.associated_entity is not None:
            tools_by_entity[tool.associated_entity].append(tool)

    return SemanticLayer(
        entities=entities,
        relations=relations,
        tools=tools,
        tools_by_relation=tools_by_relation,
        tools_by_entity=tools_by_entity,
    )


semantic_layer = build_semantic_layer(ONTOLOGY_SOURCE_PATH)

# Build ontology entity descriptions list
ontology_entities = [
    (name, ent.description) for name, ent in semantic_layer.entities.items()
]


if __name__ == "__main__":

    # 1) Find entity
    city = semantic_layer.get_entity("City")
    print(city)

    # 2) List all outgoing relations from City
    for rel in semantic_layer.list_relations_from("City"):
        print("City --", rel.name, "-->", rel.to_entity)

    # 3) Find all tools for City→Facility(has_facility)
    tools = semantic_layer.get_tools_for_relation("City", "has_facility", "Facility")
    for t in tools:
        print("tool for City.has_facility:", t.name)

    # 4) Find all entity-level tools for Container entity
    container_tools = semantic_layer.get_tools_for_entity("Container")
    for t in container_tools:
        print("entity-level tool for Container:", t.name)
