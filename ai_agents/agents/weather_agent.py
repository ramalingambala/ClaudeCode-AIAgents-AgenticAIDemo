"""Weather Agent — specialized in weather queries with persistent memory.

Remembers past weather lookups, learns city patterns, and can share
weather info with other agents.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.weather import get_weather
from ai_agents.base_agent import BaseAgent


class WeatherAgent(BaseAgent):
    name = "weather_agent"
    description = "Specialist in weather data. Remembers past lookups and city preferences."
    system_prompt = (
        "You are a weather specialist agent with persistent memory. "
        "You remember past weather queries and can refer to them. "
        "When you look up weather, note interesting patterns. "
        "If another agent asks you about weather, help them. "
        "Always use the get_weather tool to fetch data — never guess."
    )
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get weather for.",
                    }
                },
                "required": ["city"],
            },
        }
    ]
    tool_functions = {"get_weather": get_weather}

    def _learn_from_result(self, tool_name, tool_input, result):
        if tool_name == "get_weather":
            city = result.get("city", "Unknown")
            temp_c = result.get("temperature_c", "?")
            condition = result.get("condition", "?")
            self.memory.store_fact(
                f"Weather in {city}: {temp_c}°C, {condition}",
                source="get_weather tool",
            )
            # Share weather data with other agents
            self.share_info(
                f"Current weather in {city}: {temp_c}°C, {condition}"
            )
