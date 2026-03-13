"""Demo agent built with the Anthropic SDK using the direct Anthropic API (Claude Console account)."""

import json
import os

from anthropic import Anthropic
import httpx
from dotenv import load_dotenv

from tools import calculate, get_weather, search_knowledge_base

load_dotenv()

# --- Tool definitions for the Claude API ---

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city. Returns temperature, condition, and humidity.",
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
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": 'A mathematical expression string, e.g. "2 + 3 * 4".',
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_knowledge_base",
        "description": "Search an internal knowledge base for articles matching a query. Covers topics like Python, ML, APIs, Docker, and the Claude Agent SDK.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                }
            },
            "required": ["query"],
        },
    },
]

# Map tool names to their implementations
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base,
}

SYSTEM_PROMPT = """You are a helpful assistant with access to the following tools:
- get_weather: Look up current weather for any city
- calculate: Evaluate math expressions
- search_knowledge_base: Search an internal knowledge base for articles

Use the tools when appropriate to answer user questions. You can call multiple tools
if needed to fully answer a question. Always provide a clear, helpful final response."""


def run_agent(user_message: str) -> str:
    """Run the agent loop: send message -> handle tool calls -> return final response."""
    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        http_client=httpx.Client(verify=False),
    )

    messages = [{"role": "user", "content": user_message}]

    print(f"\n{'='*60}")
    print(f"User: {user_message}")
    print(f"{'='*60}")

    # Agent loop - keeps running until Claude gives a final text response
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Process each content block
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool call and collect results
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    print(f"\n-> Tool call: {tool_name}({json.dumps(tool_input)})")

                    # Execute the tool
                    func = TOOL_FUNCTIONS[tool_name]
                    result = func(**tool_input)

                    print(f"   Result: {json.dumps(result, indent=2)}")

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(result),
                        }
                    )

            # Send tool results back to Claude
            messages.append({"role": "user", "content": tool_results})

        else:
            # Claude gave a final text response - extract and return it
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            print(f"\nAssistant: {final_text}")
            return final_text


def main():
    """Run the demo agent with sample queries."""
    print("Claude Agent Demo (Direct Anthropic API)")
    print("=" * 60)

    # Demo queries that showcase the agent's capabilities
    demo_queries = [
        "What's the weather like in Tokyo?",
        "What is (15 + 27) * 3 - 12 / 4?",
        "Search for articles about Python in the knowledge base.",
        "What's the weather in London and also calculate 100 / 7?",
    ]

    print("\nChoose a mode:")
    print("1. Run demo queries automatically")
    print("2. Interactive chat mode")
    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        for query in demo_queries:
            run_agent(query)
            print()
    else:
        print("\nInteractive mode - type 'quit' to exit\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if user_input:
                run_agent(user_input)


if __name__ == "__main__":
    main()
