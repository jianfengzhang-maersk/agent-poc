# semantic_layer/tool_decorator.py

import inspect
from typing import Optional, Tuple, get_type_hints
from .tools_registry import TOOLS_REGISTRY


def semantic_tool(
    name: Optional[str] = None,
    entity: Optional[str] = None,
    relation: Optional[Tuple[str, str, str]] = None,
    description: Optional[str] = None
):
    """
    Register a semantic tool.

    Arguments:
    - name: override tool name (defaults to function name)
    - entity: primary ontology entity type (e.g. "City")
    - relation: (source, relation_name, target) tuple
                e.g. ("Facility", "hosts_event", "ContainerEvent")
    """

    # Validate metadata consistency
    if entity and relation and entity != relation[0]:
        raise ValueError(
            f"entity='{entity}' does not match relation source='{relation[0]}'"
        )

    def wrapper(fn):
        tool_name = name or fn.__name__

        sig = inspect.signature(fn)
        type_hints = get_type_hints(fn)

        # extract input schema 自动推导参数类型
        input_schema = []
        for param_name, param in sig.parameters.items():
            input_schema.append({
                "name": param_name,
                "type": str(type_hints.get(param_name, "Any")),
                "default": param.default if param.default != inspect._empty else None,
            })

        output_type = str(type_hints.get("return", "Any"))

        final_desc = description or (inspect.getdoc(fn) or "")

        # register
        TOOLS_REGISTRY[tool_name] = {
            "fn": fn,
            "input_schema": input_schema,
            "output_type": output_type,
            "entity": entity,
            "relation": relation,
            "description": final_desc,
        }

        return fn

    return wrapper
