import dspy


# ------------------------------------------------------------
# Step 1 Signature
# ------------------------------------------------------------
class QueryUnderstandingSignature(dspy.Signature):
    """
    Step 1: Extract ontology entities + high-level intent.
    """

    # ---- Inputs ----
    query: str = dspy.InputField(
        desc="User's natural language query."
    )

    ontology_entities: list[str] = dspy.InputField(
        desc=(
            "A list of ontology entity types along with their descriptions.\n"
            "Each entry is in the form: 'EntityName: Description'."
        )
    )

    # ---- Outputs ----
    entities: list[dict] = dspy.OutputField(
        desc=(
            "Entities explicitly mentioned in the query. "
            "Each must have keys: 'type' (OntologyEntityType) and 'value' (surface mention text)."
        )
    )

    intent: str = dspy.OutputField(
        desc="A short high-level intent label capturing the user's goal."
    )



# ------------------------------------------------------------
# Step 1 DSPy Module
# ------------------------------------------------------------
class QueryUnderstanding(dspy.Module):

    def __init__(self, semantic_layer):
        """
        semantic_layer: SemanticLayer instance (from semantic_layer/runtime.py)
        """
        super().__init__()
        self.semantic_layer = semantic_layer
        self.predict = dspy.ChainOfThought(QueryUnderstandingSignature)

    def forward(self, query: str, ontology_entities: list[str]) -> QueryUnderstandingSignature:

        # Pass typed list[str] into DSPy
        return self.predict(
            query=query,
            ontology_entities=ontology_entities
        )


if __name__ == "__main__":


    from agent_poc.semantic_layer.runtime import build_semantic_layer
    from agent_poc.utils.dspy_helper import DspyHelper
    
    DspyHelper.init()
    
    sl = build_semantic_layer(
        "src/agent_poc/semantic_layer/ontology.yaml",
        "src/agent_poc/semantic_layer/tools.yaml",
    )

    # Step 1 Module
    step1 = QueryUnderstanding(sl)

    # Test query
    query = "How many containers were gated out of Sydney terminal on 20July2025?"

    # Run
    res = step1(query)

    print("Entities:", res.entities)
    print("Intent:", res.intent)

