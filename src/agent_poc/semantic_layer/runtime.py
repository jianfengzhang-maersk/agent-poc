from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Any, DefaultDict
from collections import defaultdict

import yaml


RelationKey = Tuple[str, str, str]  # (from, relation_name, to)


# -------------------------
# Data model for Ontology
# -------------------------

@dataclass
class EntitySchema:
    name: str
    description: str = ""
    synonyms: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    # relationship name -> target entity name
    relationships: Dict[str, str] = field(default_factory=dict)


@dataclass
class RelationSchema:
    name: str
    from_entity: str
    to_entity: str

    @property
    def key(self) -> RelationKey:
        return (self.from_entity, self.name, self.to_entity)


# -------------------------
# Data model for Tools
# -------------------------

ToolType = Literal["relation", "entity"]


@dataclass
class RelationSemantics:
    from_entity: str
    to_entity: str
    name: str
    intent: str
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntitySemantics:
    entity: str
    intent: str


@dataclass
class ToolSchema:
    name: str
    type: ToolType
    description: str
    input_params: List[str]
    output: str
    relation_semantics: Optional[RelationSemantics] = None
    entity_semantics: Optional[EntitySemantics] = None


# -------------------------
# Semantic Layer Runtime
# -------------------------

@dataclass
class SemanticLayer:
    # ontology
    entities: Dict[str, EntitySchema]
    relations: Dict[RelationKey, RelationSchema]

    # tools
    tools: Dict[str, ToolSchema]
    tools_by_relation: DefaultDict[RelationKey, List[ToolSchema]]
    tools_by_entity: DefaultDict[str, List[ToolSchema]]

    # ---------- Query helpers ----------

    def get_entity(self, name: str) -> Optional[EntitySchema]:
        """Exact match on entity name (case sensitive)."""
        return self.entities.get(name)

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

    def get_relation(self, from_entity: str, name: str, to_entity: str) -> Optional[RelationSchema]:
        return self.relations.get((from_entity, name, to_entity))

    def get_tools_for_relation(self, from_entity: str, name: str, to_entity: str) -> List[ToolSchema]:
        return self.tools_by_relation.get((from_entity, name, to_entity), [])

    def get_tools_for_entity(self, entity_name: str) -> List[ToolSchema]:
        return self.tools_by_entity.get(entity_name, [])

    def list_relations_from(self, entity_name: str) -> List[RelationSchema]:
        """List all outgoing relations from an entity."""
        return [r for r in self.relations.values() if r.from_entity == entity_name]

    def list_relations_to(self, entity_name: str) -> List[RelationSchema]:
        """List all incoming relations to an entity."""
        return [r for r in self.relations.values() if r.to_entity == entity_name]


# -------------------------
# Loaders
# -------------------------

def load_ontology(path: str | Path) -> Tuple[Dict[str, EntitySchema], Dict[RelationKey, RelationSchema]]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if "ontology" not in data:
        raise ValueError("ontology.yaml must have top-level key 'ontology'")

    ont = data["ontology"]
    entities_def = ont.get("entities", {})

    entities: Dict[str, EntitySchema] = {}
    relations: Dict[RelationKey, RelationSchema] = {}

    for entity_name, cfg in entities_def.items():
        description = cfg.get("description", "") or ""
        synonyms = cfg.get("synonyms", []) or []
        attributes = cfg.get("attributes", []) or []
        relationships_cfg = cfg.get("relationships", {}) or {}

        entity = EntitySchema(
            name=entity_name,
            description=description,
            synonyms=list(synonyms),
            attributes=list(attributes),
            relationships=dict(relationships_cfg),
        )
        entities[entity_name] = entity

        # derive RelationSchema from relationships
        for rel_name, target_entity in relationships_cfg.items():
            rel = RelationSchema(
                name=rel_name,
                from_entity=entity_name,
                to_entity=target_entity,
            )
            relations[rel.key] = rel

    return entities, relations


def load_tools(path: str | Path) -> Dict[str, ToolSchema]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    tools_def = data.get("tools", [])
    tools: Dict[str, ToolSchema] = {}

    for t in tools_def:
        name = t["name"]
        type_: ToolType = t["type"]
        description = t.get("description", "") or ""
        input_params = t.get("input", []) or []
        output = t.get("output", "Any")

        semantics = t.get("semantics", {}) or {}
        intent = semantics.get("intent", "")

        relation_semantics: Optional[RelationSemantics] = None
        entity_semantics: Optional[EntitySemantics] = None

        if type_ == "relation":
            rel_cfg = semantics.get("relation") or {}
            filters = semantics.get("filters") or {}

            relation_semantics = RelationSemantics(
                from_entity=rel_cfg["from"],
                to_entity=rel_cfg["to"],
                name=rel_cfg["name"],
                intent=intent,
                filters=filters,
            )
        elif type_ == "entity":
            entity_name = semantics.get("entity")
            if not entity_name:
                raise ValueError(f"Tool '{name}' has type 'entity' but no 'semantics.entity' defined")
            entity_semantics = EntitySemantics(
                entity=entity_name,
                intent=intent,
            )
        else:
            raise ValueError(f"Unknown tool type for '{name}': {type_}")

        tool = ToolSchema(
            name=name,
            type=type_,
            description=description,
            input_params=list(input_params),
            output=output,
            relation_semantics=relation_semantics,
            entity_semantics=entity_semantics,
        )
        tools[name] = tool

    return tools


def build_semantic_layer(
    ontology_path: str | Path,
    tools_path: str | Path,
) -> SemanticLayer:
    # 1) Load ontology
    entities, relations = load_ontology(ontology_path)

    # 2) Load tools
    tools = load_tools(tools_path)

    # 3) Link tools ↔ relations / entities
    tools_by_relation: DefaultDict[RelationKey, List[ToolSchema]] = defaultdict(list)
    tools_by_entity: DefaultDict[str, List[ToolSchema]] = defaultdict(list)

    for tool in tools.values():
        if tool.type == "relation" and tool.relation_semantics is not None:
            key = (
                tool.relation_semantics.from_entity,
                tool.relation_semantics.name,
                tool.relation_semantics.to_entity,
            )
            tools_by_relation[key].append(tool)
        elif tool.type == "entity" and tool.entity_semantics is not None:
            ent_name = tool.entity_semantics.entity
            tools_by_entity[ent_name].append(tool)

    return SemanticLayer(
        entities=entities,
        relations=relations,
        tools=tools,
        tools_by_relation=tools_by_relation,
        tools_by_entity=tools_by_entity,
    )


if __name__ == "__main__":
    import os
    print(os.getcwd())
    sl = build_semantic_layer(
        "src/agent_poc/semantic_layer/ontology.yaml",
        "src/agent_poc/semantic_layer/tools.yaml",
    )

    # 1) Find entity
    city = sl.get_entity("City")
    print(city)

    # 2) List all outgoing relations from City
    for rel in sl.list_relations_from("City"):
        print("City --", rel.name, "-->", rel.to_entity)

    # 3) Find all tools for City→Facility(has_facility)
    tools = sl.get_tools_for_relation("City", "has_facility", "Facility")
    for t in tools:
        print("tool for City.has_facility:", t.name)

    # 4) Find all entity-level tools for Container entity
    container_tools = sl.get_tools_for_entity("Container")
    for t in container_tools:
        print("entity-level tool for Container:", t.name)
