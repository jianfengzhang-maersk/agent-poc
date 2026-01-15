from typing import List, Tuple
import dspy


# ------------------------------------------------------------
# Step 1 Signature
# ------------------------------------------------------------
class QueryUnderstandingSignature(dspy.Signature):
    """
    Step 1: Extract related entities + high-level intent.
    """

    # ---- Inputs ----
    query: str = dspy.InputField(desc="User's natural language query.")

    ontology_entities: List[Tuple[str, str]] = dspy.InputField(
        desc=("A list of ontology entity tuples in the form (EntityName, Description).")
    )

    # ---- Outputs ----
    entities: List[str] = dspy.OutputField(
        desc=(
            "Related entities mentioned in the query. It must come from the ontology_entities."
        )
    )

    intent: str = dspy.OutputField(
        desc="A short high-level intent label capturing the user's goal, less than 8 words."
    )


# ------------------------------------------------------------
# Step 1 DSPy Module
# ------------------------------------------------------------
class QueryUnderstanding(dspy.Module):
    def __init__(self):
        """Initialize module with a canonical ontology entity description list."""
        super().__init__()
        self.predict = dspy.Predict(QueryUnderstandingSignature)

    def forward(
        self,
        query: str,
        ontology_entities: List[str],
    ) -> QueryUnderstandingSignature:
        """Allow DSPy evaluators/optimizers to override ontology_entities per example."""
        return self.predict(query=query, ontology_entities=ontology_entities)


if __name__ == "__main__":
    from agent_poc.semantic_layer.engine import ontology_entities
    from agent_poc.utils.dspy_helper import DspyHelper
    import agent_poc.utils.mlflow_helper as mlflow_helper

    DspyHelper.init_kimi()
    mlflow_helper.init()

    # Step 1 Module
    qu = QueryUnderstanding()

    # Test query
    sample_query = (
        # "How many containers were gated out of Sydney terminal on 20 July 2025?"
        "Where is container TEMU9876543 located? "
    )

    # Run
    res = qu(sample_query, ontology_entities)

    print("Entities:", res.entities)
    print("Intent:", res.intent)
