import yaml
import re
from pathlib import Path


def pascal_case(name: str) -> str:
    return "".join(word.capitalize() for word in re.split(r"[_\-\s]+", name))


def python_primitive(yaml_type: str) -> str:
    mapping = {
        "string": "str",
        "integer": "int",
        "float": "float",
        "boolean": "bool",
        "datetime": "datetime.datetime",
    }
    return mapping.get(yaml_type, "Any")


class ModelGenerator:
    """Generate Python Pydantic models, including nested structs and enums."""

    # --------------------------
    # Enum generation
    # --------------------------
    def generate_enum(self, enum_name: str, spec: dict) -> str:
        class_name = pascal_case(enum_name)

        values = spec.get("values", [])
        description = spec.get("description") or f"Enum for {enum_name}"

        items = "\n".join([f"    {v.upper()} = '{v}'" for v in values])

        return f"""
class {class_name}(Enum):
    \"\"\"
    {description}
    \"\"\"
{items}
"""

    # --------------------------
    # Struct (object) generation
    # --------------------------
    def generate_struct(self, class_name: str, spec: dict, item_mode=False) -> str:
        description = spec.get("description") or (
            f"Struct type for {class_name}"
            if not item_mode
            else f"Item struct for {class_name}"
        )

        properties = spec.get("properties", {})
        lines = []

        for attr, field_spec in properties.items():
            field_type = self.resolve_field_type(attr, field_spec)
            desc = field_spec.get("description", "")
            lines.append(
                f"    {attr}: Optional[{field_type}] = Field(None, description={desc!r})"
            )

        return f"""
class {class_name}(BaseModel):
    \"\"\"
    {description}
    \"\"\"
{chr(10).join(lines)}
"""

    # --------------------------
    # Type resolver
    # --------------------------
    def resolve_field_type(self, attr: str, spec: dict) -> str:
        typ = spec.get("type")

        if typ == "enum":
            return pascal_case(attr) + "Enum"

        if typ == "array":
            item_spec = spec.get("items", {})
            inner = self.resolve_field_type(attr + "_item", item_spec)
            return f"List[{inner}]"

        if typ == "object":
            if "properties" in spec:
                return pascal_case(attr) + "Struct"
            return "Dict[str, Any]"

        return python_primitive(typ)

    # --------------------------
    # Root entity model generator
    # --------------------------
    def generate_entity_model(self, entity_name: str, entity_def: dict) -> str:
        attributes = entity_def.get("attributes", {})
        description = entity_def.get("description", "")
        entity_class_name = pascal_case(entity_name)

        nested_structs = []
        nested_enums = []

        for attr, spec in attributes.items():
            if spec.get("type") == "enum":
                nested_enums.append(self.generate_enum(attr + "_enum", spec))

            if spec.get("type") == "object" and "properties" in spec:
                struct_name = pascal_case(attr) + "Struct"
                nested_structs.append(self.generate_struct(struct_name, spec))

            if spec.get("type") == "array":
                item = spec.get("items")
                if item and item.get("type") == "object" and "properties" in item:
                    struct_name = pascal_case(attr) + "ItemStruct"
                    nested_structs.append(
                        self.generate_struct(struct_name, item, item_mode=True)
                    )

        root_fields = []
        for attr, spec in attributes.items():
            field_type = self.resolve_field_type(attr, spec)
            desc = spec.get("description", "")
            required = spec.get("primary_key", False)

            if required:
                root_fields.append(
                    f"    {attr}: {field_type} = Field(..., description={desc!r})"
                )
            else:
                root_fields.append(
                    f"    {attr}: Optional[{field_type}] = Field(None, description={desc!r})"
                )

        docstring_block = f'    """\n    {description}\n    """' if description else ""

        root_model = f"""
class {entity_class_name}(BaseModel):
{docstring_block}
{chr(10).join(root_fields)}
"""

        file_code = """
from pydantic import BaseModel, Field
from enum import Enum
import datetime
from typing import Optional, List, Dict, Any
"""

        for enum_code in nested_enums:
            file_code += "\n" + enum_code + "\n"

        for struct_code in nested_structs:
            file_code += "\n" + struct_code + "\n"

        file_code += "\n" + root_model + "\n"
        return file_code

    # --------------------------
    # Runner
    # --------------------------
    def run(self):
        src_dir = Path("src/agent_poc/semantic_layer/ontology_data")
        out_dir = Path("src/agent_poc/semantic_layer/generated_models")
        out_dir.mkdir(exist_ok=True)

        for yaml_file in src_dir.glob("*.yaml"):
            raw = yaml.safe_load(open(yaml_file))
            entity_def = raw
            entity_name = entity_def["name"]
            code = self.generate_entity_model(entity_name, entity_def)
            out_file = out_dir / f"{entity_name.lower()}.py"
            out_file.write_text(code)
            print(f"Generated: {out_file}")


if __name__ == "__main__":
    ModelGenerator().run()
