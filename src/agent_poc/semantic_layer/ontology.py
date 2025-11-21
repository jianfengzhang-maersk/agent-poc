from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


RelationKey = Tuple[str, str, str]  # (from, relation_name, to)


@dataclass
class RelationshipSpec:
    target: str
    description: str = ""


@dataclass
class EntitySchema:
    name: str
    description: str = ""
    synonyms: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    relationships: Dict[str, RelationshipSpec] = field(default_factory=dict)


@dataclass
class RelationSchema:
    name: str
    from_entity: str
    to_entity: str
    description: str = ""

    @property
    def key(self) -> RelationKey:
        return (self.from_entity, self.name, self.to_entity)


def load_ontology(path: str | Path) -> Tuple[Dict[str, EntitySchema], Dict[RelationKey, RelationSchema]]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    # Accept both previous schema (top-level 'ontology') and new root-level 'entities'
    if "ontology" in data:
        ont = data["ontology"]
    else:
        ont = data

    entities_def = ont.get("entities", {})
    if not entities_def:
        raise ValueError("ontology definition must include 'entities'")

    entities: Dict[str, EntitySchema] = {}
    relations: Dict[RelationKey, RelationSchema] = {}

    for entity_name, cfg in entities_def.items():
        description = cfg.get("description", "") or ""
        synonyms = cfg.get("synonyms", []) or []
        attributes = cfg.get("attributes", []) or []
        relationships_cfg = cfg.get("relationships", {}) or {}

        rel_specs: Dict[str, RelationshipSpec] = {}
        for rel_name, rel_cfg in relationships_cfg.items():
            if isinstance(rel_cfg, dict):
                target_entity = rel_cfg.get("target")
                rel_description = rel_cfg.get("description", "") or ""
            else:
                target_entity = rel_cfg
                rel_description = ""

            if not target_entity:
                raise ValueError(
                    f"Relationship '{rel_name}' under entity '{entity_name}' is missing a target"
                )

            rel_specs[rel_name] = RelationshipSpec(
                target=target_entity,
                description=rel_description,
            )

            rel = RelationSchema(
                name=rel_name,
                from_entity=entity_name,
                to_entity=target_entity,
                description=rel_description,
            )
            relations[rel.key] = rel

        entity = EntitySchema(
            name=entity_name,
            description=description,
            synonyms=list(synonyms),
            attributes=list(attributes),
            relationships=rel_specs,
        )
        entities[entity_name] = entity

    return entities, relations


__all__ = [
    "RelationKey",
    "RelationshipSpec",
    "EntitySchema",
    "RelationSchema",
    "load_ontology",
]


if __name__ == "__main__":

    ontology_path = "src/agent_poc/semantic_layer/ontology.yaml"
    print(f"Loading ontology from: {ontology_path}")

    entities, relations = load_ontology(ontology_path)

    print(f"Loaded entities: {len(entities)}")
    for name, entity in entities.items():
        print(f"- {name}: {entity.description}")
        for rel_name, rel in entity.relationships.items():
            print(f"    relation {rel_name} -> {rel.target}: {rel.description}")

    print(f"\nTotal relations: {len(relations)}")
    for key, rel_schema in relations.items():
        print(f"- {key}: {rel_schema.description}")
