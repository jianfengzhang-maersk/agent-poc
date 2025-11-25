import dspy
from typing import Dict, List


class StepSignature(dspy.TypedDict):
    id: int
    tool: str
    inputs: Dict[str, str]
    output: str


class PlanSignature(dspy.Signature):
    query: str = dspy.InputField()
    intent: str = dspy.InputField()
    tools: List[Dict] = dspy.InputField(
        desc="Candidate tools with schemas and semantic metadata"
    )

    steps: List[StepSignature] = dspy.OutputField(
        desc=(
            "Execution plan as ordered steps. "
            "Each step must contain: id, tool, inputs, output."
        )
    )


class HighLevelPlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(PlanSignature)

    def forward(self, query, intent, tools):
        plan = self.predict(
            query=query,
            intent=intent,
            tools=tools
        )
        return plan.steps
