import yaml
import re
from pathlib import Path
from typing import Dict, Any, List
from textwrap import indent


def pascal_case(name: str) -> str:
    return ''.join(word.capitalize() for word in re.split(r'[_\-\s]+', name))


def python_primitive(yaml_type: str) -> str:
    mapping = {
        "string": "str",
        "integer": "int",
        "float": "float",
        "boolean": "bool",
        "datetime": "datetime.datetime"
    }
    return mapping.get(yaml_type, "Any")


class ModelGenerator:
    """Generate one Python file per entity, including nested classes inline."""

    def __init__(self):
        pass

    def generate_enum(self, enum_name: str, values: List[str]) -> str:
        class_name = pascal_case(enum_name)
        items = "\n".join([f"    {v.upper()} = '{v}'" for v in values])
        return f"""
class {class_name}(Enum):
{items}
"""

    def generate_struct(self, class_name: str, properties: Dict[str, dict]) -> str:
        lines = []
        for attr, spec in properties.items():
            field_type = self.resolve_field_type(attr, spec, nested=True)
            desc = spec.get("description", "")
            optional = "Optional[" + field_type + "]"
            lines.append(f"    {attr}: {optional} = Field(None, description={desc!r})")

        return f"""
class {class_name}(BaseModel):
{chr(10).join(lines)}
"""

    def resolve_field_type(self, attr: str, spec: dict, nested=False) -> str:
        typ = spec.get("type")

        # enum
        if typ == "enum":
            return pascal_case(attr) + "Enum"

        # array
        if typ == "array":
            item_spec = spec.get("items", {})
            inner = self.resolve_field_type(attr + "_item", item_spec, nested=True)
            return f"List[{inner}]"

        # object (struct)
        if typ == "object":
            if "properties" in spec:
                struct_name = pascal_case(attr) + "Struct"
                return struct_name
            return "Dict[str, Any]"

        # primitive type
        return python_primitive(typ)

    def generate_entity_model(self, entity_name: str, entity_def: dict) -> str:
        attributes = entity_def.get("attributes", {})
        entity_class_name = pascal_case(entity_name)

        nested_structs = []
        nested_enums = []

        # First pass: collect nested objects and enums
        for attr, spec in attributes.items():

            # enum
            if spec.get("type") == "enum":
                enum_code = self.generate_enum(attr + "_enum", spec["values"])
                nested_enums.append(enum_code)

            # object → struct
            if spec.get("type") == "object" and "properties" in spec:
                struct_name = pascal_case(attr) + "Struct"
                nested_structs.append(
                    self.generate_struct(struct_name, spec["properties"])
                )

            # array of struct
            if spec.get("type") == "array":
                item = spec.get("items", {})
                if item.get("type") == "object" and "properties" in item:
                    struct_name = pascal_case(attr) + "ItemStruct"
                    nested_structs.append(
                        self.generate_struct(struct_name, item["properties"])
                    )

        # Root class
        lines = []
        for attr, spec in attributes.items():
            desc = spec.get("description", "")
            required = spec.get("primary_key", False)
            field_type = self.resolve_field_type(attr, spec)

            if required:
                lines.append(f"    {attr}: {field_type} = Field(..., description={desc!r})")
            else:
                lines.append(f"    {attr}: Optional[{field_type}] = Field(None, description={desc!r})")

        root_model = f"""
class {entity_class_name}(BaseModel):
{chr(10).join(lines)}
"""

        # Combine everything
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

    def run(self):
        src_dir = Path("src/agent_poc/semantic_layer/ontology_data")
        out_dir = Path("src/agent_poc/semantic_layer/generated_models")
        out_dir.mkdir(exist_ok=True)

        for yaml_file in src_dir.glob("*.yaml"):
            raw = yaml.safe_load(open(yaml_file))
            entity_def = raw
            entity_name = entity_def['name']
            code = self.generate_entity_model(entity_name, entity_def)
            out_file = out_dir / f"{entity_name.lower()}.py"
            out_file.write_text(code)
            print(f"Generated: {out_file}")


if __name__ == "__main__":
    ModelGenerator().run()
