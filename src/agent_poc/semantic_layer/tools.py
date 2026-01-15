import inspect
from typing import List, Optional, Tuple, get_type_hints

from agent_poc.semantic_layer.generated_models.shipment import Shipment
from agent_poc.semantic_layer.generated_models.containerevent import Containerevent
from agent_poc.semantic_layer.generated_models.container import Container
from agent_poc.semantic_layer.generated_models.facility import Facility
from agent_poc.semantic_layer.tools_registry import TOOLS_REGISTRY


def semantic_tool(
    name: Optional[str] = None,
    entity: Optional[str] = None,
    relation: Optional[Tuple[str, str, str]] = None,
    description: Optional[str] = None,
):
    """Decorator to register a semantic-layer tool."""

    if entity and relation and entity != relation[0]:
        raise ValueError(
            f"entity='{entity}' does not match relation source='{relation[0]}'"
        )

    def wrapper(fn):
        tool_name = name or fn.__name__

        sig = inspect.signature(fn)
        type_hints = get_type_hints(fn)

        input_schema = []
        for param_name, param in sig.parameters.items():
            input_schema.append(
                {
                    "name": param_name,
                    "type": str(type_hints.get(param_name, "Any")),
                    "default": param.default
                    if param.default != inspect._empty
                    else None,
                }
            )

        output_type = str(type_hints.get("return", "Any"))
        final_desc = description or (inspect.getdoc(fn) or "")

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


# -------------------------
# 1. City.has_facility → get terminals
# -------------------------
@semantic_tool(relation=("City", "has_facility", "Facility"))
def get_terminals_by_city(city_name: str) -> list[dict]:
    """Retrieve all facilities of type 'terminal' that belong to a given city."""
    pass


# -------------------------
# 2. Facility.hosts_event → get events by facility
# -------------------------
@semantic_tool(relation=("Facility", "hosts_event", "ContainerEvent"))
def get_events_by_facility(
    facility_id: str, start_date: str, end_date: str, event_type: str
) -> List[Containerevent]:
    """Query container events that occurred at a specific facility."""
    pass


# -------------------------
# 3. Container.has_event → get events by container
# -------------------------
@semantic_tool(relation=("Container", "has_event", "ContainerEvent"))
def get_events_by_container(
    container_id: str, start_date: str, end_date: str, event_type: str
) -> List[Containerevent]:
    """Query container movement events for a specific container."""
    pass


# -------------------------
# 4. Container.belongs_to → get shipment by container
# -------------------------
@semantic_tool(relation=("Container", "belongs_to", "Shipment"))
def get_shipment_by_container(container_id: str) -> Shipment:
    """Retrieve the shipment that a given container belongs to."""
    pass


# -------------------------
# 5. Shipment.has_container → get containers by shipment
# -------------------------
@semantic_tool(relation=("Shipment", "has_container", "Container"))
def get_containers_by_shipment(shipment_id: str) -> List[Container]:
    """Retrieve all containers belonging to a specific shipment."""
    pass


# -------------------------
# 6. entity-based tool → Container
# -------------------------
@semantic_tool(entity="Container")
def get_container_details(container_id: str) -> Container:
    """Retrieve metadata and physical details of a container."""
    pass


# -------------------------
# 7. entity-based tool → Shipment
# -------------------------
@semantic_tool(entity="Shipment")
def get_shipment_details(shipment_id: str) -> Shipment:
    """Retrieve metadata details of a shipment."""
    pass


# -------------------------
# 8. entity-based tool → Facility
# -------------------------
@semantic_tool(entity="Facility")
def get_facility_details(facility_id: str) -> Facility:
    """Retrieve facility metadata and related information."""
    pass


if __name__ == "__main__":
    for key, value in TOOLS_REGISTRY.items():
        print(key, value)
