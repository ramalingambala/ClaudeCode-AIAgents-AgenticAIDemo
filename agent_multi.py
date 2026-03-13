"""Multi-agent demo with parallel execution using the Anthropic API.

Architecture:
  Orchestrator Agent
      ├── Weather Agent   (runs in parallel)
      ├── Math Agent      (runs in parallel)
      └── Research Agent  (runs in parallel)

The orchestrator analyzes the user query, delegates to specialized sub-agents
that run in parallel, then synthesizes their results into a final response.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from anthropic import Anthropic
import httpx
from dotenv import load_dotenv

from tools import calculate, get_weather, search_knowledge_base

load_dotenv()


def _create_client() -> Anthropic:
    return Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        http_client=httpx.Client(verify=False),
    )


# ---------------------------------------------------------------------------
# Specialized Sub-Agents
# ---------------------------------------------------------------------------

class WeatherAgent:
    """Sub-agent specialized in weather queries."""

    name = "weather_agent"
    SYSTEM_PROMPT = (
        "You are a weather specialist agent. You have access to a weather tool. "
        "When given a weather-related query, use the tool to fetch the data and "
        "return a concise weather summary. Only handle weather-related requests."
    )
    TOOLS = [
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
    TOOL_FUNCTIONS = {"get_weather": get_weather}

    def run(self, task: str) -> dict:
        print(f"\n  [WeatherAgent] Starting: {task}")
        client = _create_client()
        messages = [{"role": "user", "content": task}]

        while True:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=self.SYSTEM_PROMPT,
                tools=self.TOOLS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        print(f"  [WeatherAgent] Tool call: {block.name}({json.dumps(block.input)})")
                        result = self.TOOL_FUNCTIONS[block.name](**block.input)
                        print(f"  [WeatherAgent] Result: {json.dumps(result, indent=2)}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                print(f"  [WeatherAgent] Done")
                return {"agent": self.name, "result": text}


class MathAgent:
    """Sub-agent specialized in mathematical calculations."""

    name = "math_agent"
    SYSTEM_PROMPT = (
        "You are a math specialist agent. You have access to a calculator tool. "
        "When given a math problem, use the tool to compute the answer and return "
        "a clear explanation. Only handle math-related requests."
    )
    TOOLS = [
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
    TOOL_FUNCTIONS = {"calculate": calculate}

    def run(self, task: str) -> dict:
        print(f"\n  [MathAgent] Starting: {task}")
        client = _create_client()
        messages = [{"role": "user", "content": task}]

        while True:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=self.SYSTEM_PROMPT,
                tools=self.TOOLS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        print(f"  [MathAgent] Tool call: {block.name}({json.dumps(block.input)})")
                        result = self.TOOL_FUNCTIONS[block.name](**block.input)
                        print(f"  [MathAgent] Result: {json.dumps(result, indent=2)}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                print(f"  [MathAgent] Done")
                return {"agent": self.name, "result": text}


class ResearchAgent:
    """Sub-agent specialized in knowledge base searches."""

    name = "research_agent"
    SYSTEM_PROMPT = (
        "You are a research specialist agent. You have access to a knowledge base search tool. "
        "When given a research query, use the tool to find relevant articles and return "
        "a concise summary of the findings. Only handle research/search requests."
    )
    TOOLS = [
        {
            "name": "search_knowledge_base",
            "description": "Search an internal knowledge base for articles matching a query.",
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
        }
    ]
    TOOL_FUNCTIONS = {"search_knowledge_base": search_knowledge_base}

    def run(self, task: str) -> dict:
        print(f"\n  [ResearchAgent] Starting: {task}")
        client = _create_client()
        messages = [{"role": "user", "content": task}]

        while True:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=self.SYSTEM_PROMPT,
                tools=self.TOOLS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        print(f"  [ResearchAgent] Tool call: {block.name}({json.dumps(block.input)})")
                        result = self.TOOL_FUNCTIONS[block.name](**block.input)
                        print(f"  [ResearchAgent] Result: {json.dumps(result, indent=2)}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                print(f"  [ResearchAgent] Done")
                return {"agent": self.name, "result": text}


# ---------------------------------------------------------------------------
# Orchestrator Agent
# ---------------------------------------------------------------------------

AGENTS = {
    "weather_agent": WeatherAgent(),
    "math_agent": MathAgent(),
    "research_agent": ResearchAgent(),
}

ORCHESTRATOR_SYSTEM_PROMPT = """You are an orchestrator agent. Your job is to analyze the user's query
and decide which specialist agent(s) should handle it.

Available agents:
- weather_agent: Handles weather-related queries (forecasts, conditions, temperature)
- math_agent: Handles mathematical calculations and expressions
- research_agent: Handles knowledge base searches and research queries

Respond ONLY with a JSON array of delegation objects. Each object must have:
- "agent": the agent name (one of: weather_agent, math_agent, research_agent)
- "task": the specific task to delegate to that agent (rephrase if needed for clarity)

Examples:
User: "What's the weather in Paris and calculate 5 * 9?"
Response: [{"agent": "weather_agent", "task": "What is the current weather in Paris?"}, {"agent": "math_agent", "task": "Calculate 5 * 9"}]

User: "Search for Docker articles"
Response: [{"agent": "research_agent", "task": "Search for articles about Docker"}]

Respond with ONLY the JSON array, no other text."""

SYNTHESIZER_SYSTEM_PROMPT = """You are a helpful assistant that synthesizes results from multiple
specialist agents into a single, coherent, and well-formatted response for the user.

You will receive the original user query and the results from each agent.
Combine them into a clear, unified response."""


def run_orchestrator(user_message: str) -> str:
    """Orchestrator: analyze query, delegate to sub-agents in parallel, synthesize results."""
    client = _create_client()

    print(f"\n{'='*60}")
    print(f"User: {user_message}")
    print(f"{'='*60}")

    # Step 1: Orchestrator decides which agents to invoke
    print("\n[Orchestrator] Analyzing query and delegating...")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=ORCHESTRATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    plan_text = "".join(b.text for b in response.content if hasattr(b, "text"))
    print(f"[Orchestrator] Delegation plan: {plan_text}")

    try:
        delegations = json.loads(plan_text)
    except json.JSONDecodeError:
        # If the orchestrator didn't return clean JSON, try to extract it
        import re
        match = re.search(r'\[.*\]', plan_text, re.DOTALL)
        if match:
            delegations = json.loads(match.group())
        else:
            return f"Error: Orchestrator could not parse the query."

    # Step 2: Execute sub-agents in parallel
    print(f"\n[Orchestrator] Dispatching {len(delegations)} agent(s) in parallel...")
    agent_results = []

    with ThreadPoolExecutor(max_workers=len(delegations)) as executor:
        futures = {}
        for delegation in delegations:
            agent_name = delegation["agent"]
            task = delegation["task"]
            agent = AGENTS.get(agent_name)
            if agent:
                future = executor.submit(agent.run, task)
                futures[future] = agent_name
            else:
                print(f"  [Orchestrator] Unknown agent: {agent_name}, skipping.")

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result()
                agent_results.append(result)
            except Exception as e:
                print(f"  [Orchestrator] {agent_name} failed: {e}")
                agent_results.append({"agent": agent_name, "result": f"Error: {e}"})

    # Step 3: Synthesize results into a final response
    print(f"\n[Orchestrator] All agents complete. Synthesizing final response...")

    synthesis_input = f"Original user query: {user_message}\n\nAgent results:\n"
    for r in agent_results:
        synthesis_input += f"\n--- {r['agent']} ---\n{r['result']}\n"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYNTHESIZER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": synthesis_input}],
    )

    final_text = "".join(b.text for b in response.content if hasattr(b, "text"))
    print(f"\nAssistant: {final_text}")
    return final_text


def main():
    """Run the multi-agent demo."""
    print("Claude Multi-Agent Demo (Parallel Execution)")
    print("=" * 60)
    print("\nArchitecture:")
    print("  Orchestrator Agent")
    print("      ├── Weather Agent   (parallel)")
    print("      ├── Math Agent      (parallel)")
    print("      └── Research Agent  (parallel)")

    demo_queries = [
        "What's the weather in Tokyo and calculate 42 * 58?",
        "Search for articles about Docker and what's the weather in London?",
        "What is (15 + 27) * 3 - 12 / 4?",
    ]

    print("\nChoose a mode:")
    print("1. Run demo queries automatically")
    print("2. Interactive chat mode")
    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        for query in demo_queries:
            run_orchestrator(query)
            print()
    else:
        print("\nInteractive mode - type 'quit' to exit\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if user_input:
                run_orchestrator(user_input)


if __name__ == "__main__":
    main()
