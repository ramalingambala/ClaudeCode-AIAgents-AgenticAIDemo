"""Math Agent — specialized in calculations with persistent memory.

Remembers past calculations so it can refer back to them and
detect patterns in what the user frequently computes.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.calculator import calculate
from ai_agents.base_agent import BaseAgent


class MathAgent(BaseAgent):
    name = "math_agent"
    description = "Specialist in mathematical calculations. Remembers past computations."
    system_prompt = (
        "You are a math specialist agent with persistent memory. "
        "You remember past calculations and can refer to previous results. "
        "If a user asks a follow-up like 'multiply that by 2', use your memory "
        "to find the previous result. Always use the calculate tool — never do mental math."
    )
    tools = [
        {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression string.",
                    }
                },
                "required": ["expression"],
            },
        }
    ]
    tool_functions = {"calculate": calculate}

    def _learn_from_result(self, tool_name, tool_input, result):
        if tool_name == "calculate" and "result" in result:
            expr = result.get("expression", "")
            val = result.get("result", "")
            self.memory.store_fact(
                f"Calculated: {expr} = {val}",
                source="calculate tool",
            )
