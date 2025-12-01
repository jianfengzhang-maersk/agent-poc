from typing import List, Dict
import dspy


class SemanticTodoPlannerSignature(dspy.Signature):
    """
    Produce a semantic-level TODO list for the user's query.

    Chain-of-thought policy:
    - First, reason step by step INTERNALLY about the user's query,
      the available entities, and the relationships between them.
    - THEN, based on that reasoning, construct the final structured outputs
      (`todo_map` and `notes`).
    - DO NOT include your full reasoning process in the outputs. Only output
      concise TODO steps and (optionally) short notes.

    Requirements:
    - Only use canonical entities and semantic relationships.
    - Each TODO step must involve exactly 1 entity, or at most 2 entities
      if they have a direct relationship as specified in `relationships`.
    - Each step involving exactly 1 entity, or at most 2 entities if they have a direct relationship,
      as specified in `relationships`.
    - Try to keep each step short and action-oriented, and avoid breaking actions on one entity into multiple steps.
    """

    query: str = dspy.InputField(desc="User's natural language query.")

    entities: List[str] = dspy.InputField(
        desc="List of canonical entities available in the semantic layer."
    )

    relationships: List[str] = dspy.InputField(
        desc="List of semantic relationships among entities."
    )

    todo_map: Dict[str, str] = dspy.OutputField(
        desc="""A map of concise TODO steps (strings) needed to answer the query at the semantic level. Key is the name task desc, and value 
        is the involved entity or entities (as a list)."""
    )

    notes: List[str] = dspy.OutputField(
        desc="A list of short notes summarizing key assumptions or semantic decisions."
    )


class SemanticTodoPlanner(dspy.Module):
    """
    Planner module that generates semantic-level TODO steps.

    This module does NOT touch physical tables, SQL, MCP, or RAG.
    It only reasons at the semantic layer and produces structured TODO steps.
    """

    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(SemanticTodoPlannerSignature)

    def forward(self, query: str, entities: List[str], relationships: List[str]):
        """
        Runs the semantic planner and returns the structured plan.

        Returns:
        - todo_map: Dict[str, List[str]]
        - notes: str
        """
        return self.planner(
            query=query,
            entities=entities,
            relationships=relationships,
        )


if __name__ == "__main__":

    from agent_poc.utils.dspy_helper import DspyHelper

    DspyHelper.init_kimi()

    entities = ["City", "Facility", "ContainerEvent", "Container"]

    relationships = [
        # Sydney -> terminals
        "City,has_facility,Facility",
        # events happening at terminal
        "Facility,has_event,ContainerEvent",
        # events associated with containers
        "ContainerEvent,of_container,Container",
    ]
    query = "How many containers were gated out of Sydney terminal on 20 July 2025"

    planner = SemanticTodoPlanner()

    plan = planner(
        query=query,
        entities=entities,
        relationships=relationships,
        
    )

    print("TODO MAP:")
    for task, ents in plan.todo_map.items():
        print(f"- {task}  =>  {ents}")

    print("\nNOTES:")
    print(plan.notes)
