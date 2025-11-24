from typing import List
import dspy
from agent_poc.semantic_layer.runtime import build_semantic_layer


semantic_layer = build_semantic_layer(
    "src/agent_poc/semantic_layer/ontology.yaml",
)

# Build ontology entity descriptions list
ontology_entities = [
    f"{name}: {ent.description or ''}" for name, ent in semantic_layer.entities.items()
]

# ------------------------------------------------------------
# Step 1 Signature
# ------------------------------------------------------------
class QueryUnderstandingSignature(dspy.Signature):
    """
    Step 1: Extract ontology entities + high-level intent.
    """

    # ---- Inputs ----
    query: str = dspy.InputField(desc="User's natural language query.")

    ontology_entities: List[str] = dspy.InputField(
        desc=(
            "A list of ontology entity types along with their descriptions.\n"
            "Each entry is in the form: 'EntityName: Description'."
        )
    )

    # ---- Outputs ----
    entities: List[dict] = dspy.OutputField(
        desc=(
            "Entities explicitly mentioned in the query. "
            "Each must have keys: 'type' (OntologyEntityType) and 'value' (surface mention text)."
        )
    )

    intent: str = dspy.OutputField(
        desc="A short high-level intent label capturing the user's goal, less than 5 words."
    )


# ------------------------------------------------------------
# Step 1 DSPy Module
# ------------------------------------------------------------
class QueryUnderstanding(dspy.Module):

    def __init__(self, ontology_entities: List[str]):
        """Initialize module with a canonical ontology entity description list."""
        super().__init__()
        self.ontology_entities = ontology_entities
        self.predict = dspy.ChainOfThought(QueryUnderstandingSignature)

    def forward(
        self,
        query: str,
        ontology_entities: List[str] | None = None,
    ) -> QueryUnderstandingSignature:
        """Allow DSPy evaluators/optimizers to override ontology_entities per example."""
        return self.predict(query=query, ontology_entities=ontology_entities)


if __name__ == "__main__":

    from agent_poc.semantic_layer.runtime import build_semantic_layer
    from agent_poc.utils.dspy_helper import DspyHelper

    DspyHelper.init()



    # Step 1 Module
    step1 = QueryUnderstanding(ontology_entities)

    # Test query
    query = "How many containers were gated out of Sydney terminal on 20July2025?"

    # Run
    res = step1(query, ontology_entities)

    print("Entities:", res.entities)
    print("Intent:", res.intent)
