import dspy
from typing import List, Dict, Tuple, Any


# -------------------------------
# Type-Aware Planner Signature
# -------------------------------

class TypeAwarePlanSignature(dspy.Signature):
    """
    Generate a type-aware execution plan based on:
    - user intent
    - semantic layer entities/relations
    - tool schemas
    - pydantic model schema (type-level structure)

    Output is a list of ordered tool calls with correctly-typed inputs.
    """

    # ----- INPUTS -----

    # original query
    query: str = dspy.InputField()

    # high-level intent from Step 1
    intent: str = dspy.InputField()

    # entities extracted in Step 1 (e.g., [{"type": "City", "value": "Sydney"}])
    extracted_entities: List[Dict[str, str]] = dspy.InputField()

    # list[ToolInfo] (converted to serializable dict)
    # each tool includes: name, input_schema, output_type, associated_entity/relation
    candidate_tools: List[Dict[str, Any]] = dspy.InputField(
        desc="Tools relevant to this query, including argument schema and return types."
    )

    # list[EntitySchema] (converted to dict): name, description, synonyms, relationships
    entity_schemas: List[Dict[str, Any]] = dspy.InputField(
        desc="Semantic-layer entity definitions (semantic, not type-level)."
    )

    # list[RelationSchema] (converted to dict)
    relations: List[Dict[str, Any]] = dspy.InputField(
        desc="Relevant relations between entities for reasoning about tool ordering."
    )

    # mapping: entity_name → {fields...}  (via model.model_json_schema())
    model_schemas: Dict[str, Dict[str, Any]] = dspy.InputField(
        desc="Type-level field schemas from Pydantic models."
    )


    # ----- OUTPUTS -----

    # Ordered execution plan
    # Each step contains:
    #   id
    #   tool
    #   inputs: {arg_name: literal | variable | field_path}
    #   output: variable name bound to tool's result
    steps: List[Dict[str, Any]] = dspy.OutputField(
        desc="Ordered type-aware plan steps using tools and entity schemas.")



# agent_poc/nodes/planning/type_aware_planner.py

from typing import Any, Dict, List

import dspy


class TypeAwarePlanner(dspy.Module):
    """
    Step 3: Type-aware high-level planner.

    Given:
      - query + intent + extracted_entities
      - candidate_tools (ToolInfo → dict)
      - entity_schemas (EntitySchema → dict)
      - relations (RelationSchema → dict)
      - model_schemas (Pydantic models → json schema)

    Produce:
      - an ordered list of tool calls (steps), where each step:
        * chooses a tool
        * wires inputs using literals or previous step outputs / fields
        * assigns an output variable
    """

    def __init__(self) -> None:
        super().__init__()
        # 用 CoT，让 LLM 显式走推理过程
        self.planner = dspy.ChainOfThought(TypeAwarePlanSignature)

    def forward(
        self,
        query: str,
        intent: str,
        extracted_entities: List[Dict[str, str]],
        candidate_tools: List[Dict[str, Any]],
        entity_schemas: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        model_schemas: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run the type-aware planner and return a dict with a 'steps' field.

        上层可以直接用 result.steps 拿到 plan。
        """

        # 这里不做太多 pre-processing，把结构尽量原样交给 LLM，
        # 这样方便你后面在 prompt 里观察和调整。
        result = self.planner(
            query=query,
            intent=intent,
            extracted_entities=extracted_entities,
            candidate_tools=candidate_tools,
            entity_schemas=entity_schemas,
            relations=relations,
            model_schemas=model_schemas,
        )

        # DSPy 会把输出封装在 result.steps 里
        return {"steps": result.steps}


if __name__ == "__main__":
    from agent_poc.semantic_layer.engine import build_semantic_layer
    from agent_poc.modules.planning.planner import TypeAwarePlanner
    from agent_poc.modules.planning.adapters import (
        tools_to_candidate_tools,
        entities_to_schemas,
        relations_to_schemas,
        models_to_schemas,
    )
    from agent_poc.semantic_layer.generated_models.container import Container
    from agent_poc.semantic_layer.generated_models.shipment import Shipment
    from agent_poc.semantic_layer.generated_models.facility import Facility
    from agent_poc.semantic_layer.generated_models.containerevent import Containerevent
    from agent_poc.semantic_layer.engine import semantic_layer
    from agent_poc.utils.dspy_helper import DspyHelper
    
    
    import dspy

    DspyHelper.init_kimi()
    
    # 假设你已经在别处配置好了 dspy.settings.lm = ...
    planner = TypeAwarePlanner()

    # 1) semantic layer
    sl = semantic_layer

    # 2) 上游 step 1 / 2 / 3.1 的结果（这里简化）
    query = "How many containers were gated out of Sydney terminal on 20July2025?"
    intent = "container_event_count"
    extracted_entities = [
        {"type": "City", "value": "Sydney"},
        {"type": "ContainerEvent", "value": "gated out"},
    ]

    # 3) 选出来的 candidate tools（你已有逻辑）
    relation_tools = sl.get_tools_for_relation("City", "has_facility", "Facility") + \
                    sl.get_tools_for_relation("Facility", "hosts_event", "ContainerEvent")
    # 再加上一个通用 count 工具
    # ...

    candidate_tools = tools_to_candidate_tools(relation_tools)

    # 4) entity & relation schemas（只选相关的）
    entity_schemas = entities_to_schemas(
        [sl.entities["City"], sl.entities["Facility"], sl.entities["ContainerEvent"]]
    )
    relations = relations_to_schemas(sl.relations.values())

    # 5) Pydantic model schemas（只给相关 entity）
    models = {
        "Facility": Facility,
        "ContainerEvent": Containerevent,
        "Container": Container,
        "Shipment": Shipment,
    }
    model_schemas = models_to_schemas(models)

    # 6) 调 planner
    result = planner(
        query=query,
        intent=intent,
        extracted_entities=extracted_entities,
        candidate_tools=candidate_tools,
        entity_schemas=entity_schemas,
        relations=relations,
        model_schemas=model_schemas,
    )

    plan_steps = result["steps"]
    for index, step in enumerate(plan_steps):
         print(f"Step {index + 1}: {step}")
