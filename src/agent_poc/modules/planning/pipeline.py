"""Planning pipeline that composes type-aware planner and code generation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent_poc.modules.planning.plan_generation import TypeAwarePlanner
from agent_poc.modules.planning.code_generation import PythonCodeGen
from agent_poc.modules.planning.schema_converters import (
    tools_to_dict,
    entities_to_dict,
    relations_to_dict,
    models_to_dict,
)
from agent_poc.semantic_layer.ontology import RelationKey
from agent_poc.semantic_layer.engine import semantic_layer


def run_planning(
    query: str,
    intent: str,
    extracted_entities: List[dict],
    expanded_entities: List[str],
    active_relations: List[RelationKey],
    planner: TypeAwarePlanner | None = None,
    codegen: PythonCodeGen | None = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Pipeline Step 3: Type-aware planning + code generation.

    Args:
        query: Original user query
        intent: High-level intent from query understanding
        extracted_entities: Entities extracted in step 1
        expanded_entities: Entity types from semantic grounding
        active_relations: Active relations from semantic grounding
        planner: Optional planner instance
        codegen: Optional code generator instance

    Returns:
        Tuple of (plan_steps, python_code)
    """

    if not extracted_entities or not expanded_entities:
        return [], ""

    planner = planner or TypeAwarePlanner()
    codegen = codegen or PythonCodeGen()

    # Step 3.1: Collect candidate tools based on active relations
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

    # Convert to dict format for LLM
    candidate_tools = tools_to_dict(unique_tools)

    # Step 3.2: Prepare entity schemas for expanded entities
    entity_schemas = entities_to_dict(
        [semantic_layer.entities[e] for e in expanded_entities]
    )

    # Step 3.3: Prepare relation schemas for active relations
    relation_schemas = relations_to_dict(
        [semantic_layer.relations[rel] for rel in active_relations]
    )

    # Step 3.4: Prepare Pydantic model schemas
    # Import models dynamically based on expanded entities
    from agent_poc.semantic_layer.generated_models.container import Container
    from agent_poc.semantic_layer.generated_models.shipment import Shipment
    from agent_poc.semantic_layer.generated_models.facility import Facility
    from agent_poc.semantic_layer.generated_models.containerevent import Containerevent
    from agent_poc.semantic_layer.generated_models.city import City

    model_mapping = {
        "Container": Container,
        "Shipment": Shipment,
        "Facility": Facility,
        "ContainerEvent": Containerevent,
        "City": City,
    }

    # Only include models for expanded entities
    models = {
        name: model_cls
        for name, model_cls in model_mapping.items()
        if name in expanded_entities
    }
    model_schemas = models_to_dict(models)

    # Step 3.5: Generate execution plan
    planner_result = planner(
        query=query,
        intent=intent,
        extracted_entities=extracted_entities,
        candidate_tools=candidate_tools,
        entity_schemas=entity_schemas,
        relations=relation_schemas,
        model_schemas=model_schemas,
    )

    plan_steps = planner_result["steps"]

    # Step 3.6: Generate Python code from plan
    # Build tools dict for codegen
    tools_dict = {tool["name"]: tool for tool in candidate_tools}

    codegen_result = codegen(
        plan=plan_steps,
        tools=tools_dict,
        model_schemas=model_schemas,
    )

    python_code = codegen_result["python_code"]

    return plan_steps, python_code


if __name__ == "__main__":
    from agent_poc.utils.dspy_helper import DspyHelper
    from agent_poc.modules.query_understanding.query_understanding import (
        QueryUnderstanding,
    )
    from agent_poc.modules.semantic_grounding.pipeline import run_semantic_grounding
    from agent_poc.modules.semantic_grounding.relation_filtering import (
        RelationFiltering,
    )
    from agent_poc.semantic_layer.engine import semantic_layer, ontology_entities
    import agent_poc.utils.mlflow_helper as mlfow_helper

    import json

    mlfow_helper.init()
    # Initialize DSPy
    DspyHelper.init_kimi()

    print("=" * 80)
    print("Complete Pipeline: Query Understanding → Semantic Grounding → Planning")
    print("=" * 80)

    # Sample query
    sample_query = (
        "How many containers were gated out of Sydney terminal on 20 July 2025?"
    )
    print(f"\nQuery: {sample_query}")

    # ============================================================
    # Step 1: Query Understanding
    # ============================================================
    print("\n" + "-" * 80)
    print("Step 1: Query Understanding")
    print("-" * 80)

    query_understanding = QueryUnderstanding(ontology_entities)
    query_understanding.load(
        "src/agent_poc/modules/query_understanding/query_understanding_optimized_2.json"
    )

    qu_result = query_understanding(
        query=sample_query, ontology_entities=ontology_entities
    )

    print("\nExtracted entities:")
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
    print("\nActive relations:")
    for rel in active_relations:
        print(f"  {rel[0]} --{rel[1]}--> {rel[2]}")

    # ============================================================
    # Step 3: Planning (Planner + CodeGen)
    # ============================================================
    print("\n" + "-" * 80)
    print("Step 3: Planning (Planner + CodeGen)")
    print("-" * 80)

    plan_steps, python_code = run_planning(
        query=sample_query,
        intent=qu_result.intent,
        extracted_entities=qu_result.entities,
        expanded_entities=expanded_entities,
        active_relations=active_relations,
    )

    # ============================================================
    # Output: Execution Plan and Generated Code
    # ============================================================
    print("\n" + "=" * 80)
    print("Generated Execution Plan")
    print("=" * 80)

    for step in plan_steps:
        print(f"\nStep {step['id']}: {step['tool']}")
        print(f"  Inputs: {json.dumps(step['inputs'], indent=4)}")
        print(f"  Output: {step['output']}")

    print("\n" + "=" * 80)
    print("Generated Python Code")
    print("=" * 80)
    print()
    print(python_code)

    print("\n" + "=" * 80)
    print("✓ Complete pipeline executed successfully!")
    print(f"  Total steps in plan: {len(plan_steps)}")
    print(f"  Generated code lines: {len(python_code.splitlines())}")
    print("=" * 80)
