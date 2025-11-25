from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _extract_entities_from_payload(payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Normalize various ontology payload shapes into an entity dictionary."""

    if not payload:
        return {}

    data = payload
    if "ontology" in data:
        data = data.get("ontology") or {}

    if "entities" in data:
        entities = data.get("entities") or {}
        if not isinstance(entities, dict):
            raise ValueError(f"'entities' section in {source} must be a mapping")
        return entities

    if "name" in data:
        entity_name = data["name"]
        if not entity_name:
            raise ValueError(f"Entity definition in {source} must include a non-empty name")
        cfg = {k: v for k, v in data.items() if k != "name"}
        return {entity_name: cfg}

    if len(data) == 1:
        (entity_name, cfg), = data.items()
        if isinstance(entity_name, str) and isinstance(cfg, dict):
            return {entity_name: cfg}

    raise ValueError(
        f"Unable to extract entity definition from {source}. "
        "Provide either an 'entities' mapping or a single entity document."
    )


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

    if path.is_dir():
        entities_payload: Dict[str, Any] = {}
        yaml_files = sorted(
            [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}]
        )

        if not yaml_files:
            raise ValueError(f"ontology directory '{path}' does not contain any YAML files")

        for yaml_path in yaml_files:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            extracted = _extract_entities_from_payload(raw, str(yaml_path))
            for entity_name, cfg in extracted.items():
                if entity_name in entities_payload:
                    raise ValueError(
                        f"Duplicate entity '{entity_name}' found while loading {yaml_path}"
                    )
                entities_payload[entity_name] = cfg

        data = {"entities": entities_payload}
        ont = data
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        # Accept both previous schema (top-level 'ontology') and new root-level 'entities'
        if "ontology" in data:
            ont = data["ontology"] or {}
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

    ontology_path = Path(__file__).with_name("ontology_data")
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
