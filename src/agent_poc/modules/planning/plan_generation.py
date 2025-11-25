import dspy
from typing import List, Dict, Tuple, Any


# -------------------------------
# Type-Aware Planner Signature
# -------------------------------

class TypeAwarePlanSignature(dspy.Signature):
    """
    You are a Type-Aware Planner that generates an ordered sequence of tool calls.
    Follow ALL rules strictly:

    =====================
    ## DATE RULES
    =====================
    1. All dates MUST be in format "YYYY-MM-DD".
    2. NEVER include time, timezone, or T/Z suffix.

    =====================
    ## ARRAY EXPANSION RULES
    =====================
    1. NEVER produce fields such as "iterate_over", "loop", "map", "for_each".
    2. Loops belong ONLY to code generation, NOT this planner.
    3. If a tool needs to be applied to multiple items, ALWAYS use:
           <list_var>[*].<field>
       Example:
           terminals[*].facility_id

    =====================
    ## TOOL USAGE RULES
    =====================
    1. Each step MUST call exactly one tool.
    2. Use ONLY the tools provided in candidate_tools.
    3. Do NOT invent new tools or new arguments.
    4. Inputs to tools must be:
        - literals extracted from the query, or
        - variables produced by earlier steps, or
        - field paths (including nested) of earlier outputs.

    =====================
    ## FIELD-PATH RULES
    =====================
    Examples of allowed field paths:
        "terminals[0].facility_id"
        "terminals[*].facility_id"
        "events[*].container_id"
        "shipment.containers[*].container_number"

    =====================
    ## OUTPUT FORMAT RULES
    =====================
    MUST output a list of steps:
    [
      {
        "id": 1,
        "tool": "get_xxx",
        "inputs": {...},
        "output": "var_name"
      }
    ]
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
        # Use CoT to let LLM explicitly walk through reasoning process
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
        # Minimal pre-processing - pass structures as-is to LLM
        # This makes it easier to observe and adjust prompts later
        result = self.planner(
            query=query,
            intent=intent,
            extracted_entities=extracted_entities,
            candidate_tools=candidate_tools,
            entity_schemas=entity_schemas,
            relations=relations,
            model_schemas=model_schemas
        )

        # DSPy wraps the output in result.steps
        return {"steps": result.steps}


if __name__ == "__main__":
    from agent_poc.utils.dspy_helper import DspyHelper
    from agent_poc.modules.query_understanding.query_understanding import QueryUnderstanding
    from agent_poc.modules.semantic_grounding.pipeline import run_semantic_grounding
    from agent_poc.modules.semantic_grounding.relation_filtering import RelationFiltering
    from agent_poc.semantic_layer.engine import semantic_layer, ontology_entities
    from agent_poc.modules.planning.schema_converters import (
        tools_to_dict,
        entities_to_dict,
        relations_to_dict,
        models_to_dict,
    )
    from agent_poc.semantic_layer.generated_models.container import Container
    from agent_poc.semantic_layer.generated_models.shipment import Shipment
    from agent_poc.semantic_layer.generated_models.facility import Facility
    from agent_poc.semantic_layer.generated_models.containerevent import Containerevent
    import json

    # Initialize DSPy
    DspyHelper.init_kimi()
    
    print("=" * 80)
    print("Complete Pipeline: Query Understanding → Semantic Grounding → Planning")
    print("=" * 80)
    
    # Sample query
    sample_query = "How many containers were gated out of Sydney terminal on 20 July 2025?"
    print(f"\nQuery: {sample_query}")
    
    # ============================================================
    # Step 1: Query Understanding
    # ============================================================
    print("\n" + "-" * 80)
    print("Step 1: Query Understanding")
    print("-" * 80)
    
    query_understanding = QueryUnderstanding(ontology_entities)
    query_understanding.load("src/agent_poc/modules/query_understanding/query_understanding_optimized_2.json")
    
    qu_result = query_understanding(query=sample_query, ontology_entities=ontology_entities)
    
    print(f"\nExtracted entities:")
    for entity in qu_result.entities:
        print(f"  - {entity['type']}: {entity['value']}")
    print(f"\nIntent: {qu_result.intent}")
    
    # ============================================================
    # Step 2: Semantic Grounding
    # ============================================================
    print("\n" + "-" * 80)
    print("Step 2: Semantic Grounding")
    print("-" * 80)
    
    filtering_model = RelationFiltering(batch_size=4)
    expanded_entities, active_relations = run_semantic_grounding(
        query=sample_query,
        entities=qu_result.entities,
        intent=qu_result.intent,
        filtering_model=filtering_model,
    )
    
    # Filter to valid entities only
    expanded_entities = [
        entity for entity in expanded_entities if semantic_layer.has_entity(entity)
    ]
    
    print(f"\nExpanded entities: {expanded_entities}")
    print(f"\nActive relations:")
    for rel in active_relations:
        print(f"  {rel[0]} --{rel[1]}--> {rel[2]}")
    
    # ============================================================
    # Step 3: Planning
    # ============================================================
    print("\n" + "-" * 80)
    print("Step 3: Type-Aware Planning")
    print("-" * 80)
    
    # 3.1) Collect candidate tools based on active relations
    candidate_tools_list = []
    for rel in active_relations:
        tools = semantic_layer.get_tools_for_relation(rel[0], rel[1], rel[2])
        candidate_tools_list.extend(tools)
    
    # Add entity-level tools for expanded entities
    for entity_name in expanded_entities:
        entity_tools = semantic_layer.get_tools_for_entity(entity_name)
        candidate_tools_list.extend(entity_tools)
    
    # Remove duplicates by tool name
    seen_tools = set()
    unique_tools = []
    for tool in candidate_tools_list:
        if tool.name not in seen_tools:
            seen_tools.add(tool.name)
            unique_tools.append(tool)
    
    candidate_tools = tools_to_dict(unique_tools)
    print(f"\nCandidate tools ({len(candidate_tools)}):")
    for tool in candidate_tools:
        print(f"  - {tool['name']}")
    
    # 3.2) Prepare entity schemas for expanded entities
    entity_schemas = entities_to_dict([
        semantic_layer.entities[e] for e in expanded_entities
    ])
    
    # 3.3) Prepare relation schemas for active relations
    relation_schemas = relations_to_dict([
        semantic_layer.relations[rel] for rel in active_relations
    ])
    
    # 3.4) Prepare Pydantic model schemas
    models = {
        "Facility": Facility,
        "ContainerEvent": Containerevent,
        "Container": Container,
        "Shipment": Shipment,
    }
    model_schemas = models_to_dict(models)
    
    # 3.5) Run planner
    planner = TypeAwarePlanner()
    result = planner(
        query=sample_query,
        intent=qu_result.intent,
        extracted_entities=qu_result.entities,
        candidate_tools=candidate_tools,
        entity_schemas=entity_schemas,
        relations=relation_schemas,
        model_schemas=model_schemas,
    )
    
    # ============================================================
    # Output: Execution Plan
    # ============================================================
    print("\n" + "=" * 80)
    print("Generated Execution Plan")
    print("=" * 80)
    
    plan_steps = result["steps"]
    for step in plan_steps:
        print(f"\nStep {step['id']}: {step['tool']}")
        print(f"  Inputs: {json.dumps(step['inputs'], indent=4)}")
        print(f"  Output: {step['output']}")
    
    print("\n" + "=" * 80)
    print(f"✓ Complete pipeline executed successfully!")
    print(f"  Total steps in plan: {len(plan_steps)}")
    print("=" * 80)
