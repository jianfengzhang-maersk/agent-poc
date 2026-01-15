import ast
import re
from typing import Any, Dict, List

import dspy


class PythonCodeGenSignature(dspy.Signature):
    """
    Convert a high-level execution plan (a list of tool-call steps)
    into fully executable Python code.

    ============================
    STRICT OUTPUT RULES
    ============================
    1. MUST output ONLY valid Python code.
       - No comments.
       - No markdown.
       - No explanation.
       - No surrounding quotes or backticks.

    2. MUST generate a function:
           def run():
               ...

    3. MUST import required tool functions at the top:
           from agent_poc.semantic_layer.tools import <tool names>

       If multiple tools are used:
           from agent_poc.semantic_layer.tools import (
               tool1,
               tool2,
               ...
           )

    4. MUST respect all tool signatures exactly as provided in `tools`.
       - MUST NOT invent arguments.
       - MUST NOT invent tools.
       - MUST NOT modify parameter names.

    5. MUST expand any list pattern `<var>[*].field` using Python loops:
           results = []
           for item in <var>:
               r = tool(field=item.field, ...)
               results.extend(r)
       The final collected object is assigned to the step's `output`.

    6. Every step MUST be rendered in order:
           step1_output = tool(...)
           step2_output = ...
       Your code MUST use the exact `output` variable name defined in the plan steps.

    7. Literals extracted from the query must be inserted exactly as provided
       (e.g., "2025-07-20").

    8. The final line of run() MUST return the last step's output variable.

    ============================
    INPUT FORMAT
    ============================
    - plan: list of steps. Each step has fields:
        {
          "id": int,
          "tool": str,
          "inputs": { param: literal_or_var_or_fieldpath },
          "output": str
        }

    - tools: dict { tool_name: ToolInfo-like dict }
      Must be used to look up official parameter names.

    - model_schemas: dict of Pydantic model JSON schemas
      (You may ignore these unless needed for field reference resolution.)

    ============================
    PYTHON CODE STYLE
    ============================
    - Use standard Python 3.
    - Use explicit for-loops; do NOT use list comprehensions for side-effects.
    - No type hints required.
    - No comments.
    - No blank lines at the end.
    - Clean, minimal, deterministic code.

    """

    plan: List[Dict[str, Any]] = dspy.InputField()
    tools: Dict[str, Any] = dspy.InputField()
    model_schemas: Dict[str, Any] = dspy.InputField()

    python_code: str = dspy.OutputField(
        desc="Executable Python code implementing a function run()"
    )


class PythonCodeGen(dspy.Module):
    """
    Step 4: Turn a high-level execution plan into concrete Python code.

    Usage:
        codegen = PythonCodeGen()
        result = codegen(plan=plan_steps, tools=tools_dict, model_schemas=model_schemas)
        python_code = result["python_code"]

    This module does not depend on the concrete ToolInfo structure as long as `tools`
    is a JSON-serializable dict (e.g., convert ToolInfo to dict in the adapter layer).
    """

    def __init__(
        self,
        validate_syntax: bool = True,
        strip_markdown_fences: bool = True,
    ) -> None:
        super().__init__()
        self._validate_syntax = validate_syntax
        self._strip_markdown_fences = strip_markdown_fences

        # Use ChainOfThought so the LLM reasons internally before emitting python_code
        self.generator = dspy.ChainOfThought(PythonCodeGenSignature)

    # dspy.Module requirement: forward must return dict/object whose fields match the Signature outputs
    def forward(
        self,
        plan: List[Dict[str, Any]],
        tools: Dict[str, Any],
        model_schemas: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        plan: planner-generated step list (JSON DSL)
        tools: dict mapping tool name to its schema (handlers should be stripped, metadata only)
        model_schemas: optional Pydantic model JSON schemas for LLM field reference
        """

        # 1) Invoke LLM to generate code
        result = self.generator(
            plan=plan,
            tools=tools,
            model_schemas=model_schemas,
        )

        code = result.python_code

        # 2) Defensive post-process to strip markdown fences such as ```python
        code = self._postprocess_code(code)

        # 3) Optional syntax validation to catch errors early
        if self._validate_syntax:
            self._assert_valid_python(code)

        return {"python_code": code}

    # ------------------- Internal helpers -------------------

    def _postprocess_code(self, code: str) -> str:
        """Clean up LLM output by stripping fences or extra whitespace."""
        if not isinstance(code, str):
            code = str(code)

        code = code.strip()

        if self._strip_markdown_fences:
            # Typical fences: ```python ... ```, ``` ... ```
            fence_pattern = r"```(?:python)?\s*(.*?)```"
            m = re.search(fence_pattern, code, flags=re.DOTALL | re.IGNORECASE)
            if m:
                code = m.group(1).strip()

        # Strip once more to remove any residual whitespace
        return code.strip()

    def _assert_valid_python(self, code: str) -> None:
        """Use ast.parse to verify whether the code is valid Python."""
        try:
            ast.parse(code, mode="exec")
        except SyntaxError as e:
            # Raising ValueError keeps the failure visible to the caller
            raise ValueError(f"Generated code is not valid Python: {e}") from e


if __name__ == "__main__":
    # ---- Step 1: planner output ----
    from agent_poc.utils.dspy_helper import DspyHelper

    DspyHelper.init_kimi()

    plan_steps = [
        {
            "id": 1,
            "tool": "get_terminals_by_city",
            "inputs": {"city_name": "Sydney"},
            "output": "terminals",
        },
        {
            "id": 2,
            "tool": "get_events_by_facility",
            "inputs": {
                "facility_id": "terminals[*].facility_id",
                "start_date": "2025-07-20",
                "end_date": "2025-07-20",
                "event_type": "gate_out",
            },
            "output": "events",
        },
    ]

    # ---- Step 2: mock tools ----

    tools = {
        "get_terminals_by_city": {
            "name": "get_terminals_by_city",
            "input_schema": [
                {"name": "city_name", "type": "string"},
            ],
            "output_type": "List[Facility]",
        },
        "get_events_by_facility": {
            "name": "get_events_by_facility",
            "input_schema": [
                {"name": "facility_id", "type": "string"},
                {"name": "start_date", "type": "string"},
                {"name": "end_date", "type": "string"},
                {"name": "event_type", "type": "string"},
            ],
            "output_type": "List[ContainerEvent]",
        },
    }

    model_schemas = {}  # not needed

    # ---- Step 3: CodeGen ----

    cg = PythonCodeGen()
    cg_result = cg(plan=plan_steps, tools=tools, model_schemas=model_schemas)

    print(cg_result["python_code"])
