"""Orchestrator Agent — the brain of the multi-agent system.

Responsibilities:
1. Analyze user queries and create execution plans
2. Delegate tasks to specialized agents in parallel
3. Monitor agent status and re-plan on failures
4. Synthesize results using inter-agent communication
5. Maintain goals and track progress across sessions
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from anthropic import Anthropic
import httpx

from ai_agents.memory.store import MemoryStore
from ai_agents.communication.message_bus import MessageBus
from ai_agents.agents.weather_agent import WeatherAgent
from ai_agents.agents.math_agent import MathAgent
from ai_agents.agents.research_agent import ResearchAgent


def _create_client() -> Anthropic:
    return Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        http_client=httpx.Client(verify=False),
    )


PLANNER_PROMPT = """You are an orchestrator agent that plans and delegates tasks.

You have persistent memory of past interactions and you manage these specialized agents:
- weather_agent: Weather data lookups for any city
- math_agent: Mathematical calculations and expressions
- research_agent: Knowledge base searches for articles

IMPORTANT RULES:
1. Analyze the user query and decide which agent(s) to invoke.
2. If the query references past results (e.g., "what was the weather last time?"),
   check the memory context provided to you.
3. If multiple agents are needed, they will run IN PARALLEL.

Respond ONLY with a JSON object:
{
  "plan": "Brief description of what you'll do",
  "delegations": [
    {"agent": "agent_name", "task": "specific task description"},
    ...
  ],
  "goals": ["any new goals to track"]
}

If the user is just chatting or asking about memory/history, respond with:
{
  "plan": "Direct response — no delegation needed",
  "delegations": [],
  "goals": [],
  "direct_response": "Your response here"
}"""

SYNTHESIZER_PROMPT = """You are an orchestrator that synthesizes results from multiple AI agents.

You have memory of past interactions. Use it to provide continuity in your responses.
Combine the agent results into a clear, unified, well-formatted response.
If an agent failed, acknowledge it gracefully and suggest alternatives.
Reference past interactions from memory when relevant."""


class Orchestrator:
    """Goal-driven orchestrator with memory, planning, and agent coordination."""

    def __init__(self):
        self.bus = MessageBus()
        self.memory = MemoryStore("orchestrator")
        self.agents = {
            "weather_agent": WeatherAgent(self.bus),
            "math_agent": MathAgent(self.bus),
            "research_agent": ResearchAgent(self.bus),
        }
        self._session_start = time.time()

    def _build_planner_context(self, user_message: str) -> str:
        """Build the planner prompt with memory context."""
        parts = [PLANNER_PROMPT]

        # Add orchestrator's memory
        recent = self.memory.get_recent_interactions(limit=5)
        if recent:
            parts.append("\n## Past Interactions (your memory):")
            for interaction in recent:
                parts.append(
                    f"- Query: \"{interaction['user_query'][:100]}\" "
                    f"-> Tools/Agents used: {interaction['tools_used']}"
                )

        # Add known facts from all agents
        parts.append("\n## Known Facts from Agents:")
        for agent_id, agent in self.agents.items():
            facts = agent.memory.recall_facts(limit=5)
            if facts:
                parts.append(f"\n  {agent_id}:")
                for f in facts:
                    parts.append(f"    - {f['fact']}")

        # Add active goals
        goals = self.memory.get_active_goals()
        if goals:
            parts.append("\n## Active Goals:")
            for g in goals:
                parts.append(f"- {g['goal']}")

        # Add any inter-agent messages
        all_msgs = self.bus.get_all_messages()
        if all_msgs:
            parts.append(f"\n## Recent Agent Communications ({len(all_msgs)} messages)")

        return "\n".join(parts)

    def run(self, user_message: str) -> str:
        """Full orchestration: plan -> delegate (parallel) -> synthesize."""
        client = _create_client()

        print(f"\n{'='*60}")
        print(f"User: {user_message}")
        print(f"{'='*60}")

        # Step 1: Plan
        print("\n[Orchestrator] Planning...")
        planner_context = self._build_planner_context(user_message)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=planner_context,
            messages=[{"role": "user", "content": user_message}],
        )

        plan_text = "".join(b.text for b in response.content if hasattr(b, "text"))
        print(f"[Orchestrator] Plan: {plan_text}")

        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if match:
                plan = json.loads(match.group())
            else:
                return "Error: Could not parse the plan."

        # Handle direct responses (no delegation needed)
        if plan.get("direct_response"):
            result = plan["direct_response"]
            print(f"\nAssistant: {result}")
            self.memory.store_interaction(user_message, result, ["direct"])
            return result

        # Track new goals
        for goal in plan.get("goals", []):
            self.memory.add_goal(goal)
            print(f"[Orchestrator] New goal: {goal}")

        delegations = plan.get("delegations", [])
        if not delegations:
            result = plan.get("plan", "I'm not sure how to help with that.")
            print(f"\nAssistant: {result}")
            self.memory.store_interaction(user_message, result, ["none"])
            return result

        # Step 2: Delegate to agents in parallel
        print(f"\n[Orchestrator] Dispatching {len(delegations)} agent(s) in parallel...")
        agent_results = []
        agents_used = []

        with ThreadPoolExecutor(max_workers=len(delegations)) as executor:
            futures = {}
            for delegation in delegations:
                agent_name = delegation["agent"]
                task = delegation["task"]
                agent = self.agents.get(agent_name)
                if agent:
                    agents_used.append(agent_name)
                    future = executor.submit(agent.run, task)
                    futures[future] = agent_name
                else:
                    print(f"  [Orchestrator] Unknown agent: {agent_name}")

            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result()
                    agent_results.append(result)

                    # Step 2b: Monitor — check if agent failed, re-plan
                    if result.get("status") == "error":
                        print(f"  [Orchestrator] {agent_name} FAILED. Checking for fallback...")
                        self.bus.send(
                            "orchestrator", agent_name,
                            "Your task failed. Please retry or simplify.",
                            msg_type="request",
                        )
                except Exception as e:
                    print(f"  [Orchestrator] {agent_name} exception: {e}")
                    agent_results.append({
                        "agent": agent_name,
                        "status": "error",
                        "result": str(e),
                    })

        # Step 3: Synthesize
        print(f"\n[Orchestrator] Synthesizing final response...")

        # Build synthesis context with memory
        synth_parts = [f"Original user query: {user_message}"]

        # Include memory context
        recent = self.memory.get_recent_interactions(limit=3)
        if recent:
            synth_parts.append("\nPast interactions for context:")
            for interaction in recent:
                synth_parts.append(f"- Previously asked: \"{interaction['user_query'][:80]}\"")

        synth_parts.append("\nAgent results:")
        for r in agent_results:
            status = r.get("status", "unknown")
            synth_parts.append(f"\n--- {r['agent']} (status: {status}) ---")
            synth_parts.append(r["result"])

        # Include inter-agent comms
        messages = self.bus.get_all_messages()
        if messages:
            synth_parts.append("\nInter-agent communications:")
            for m in messages[-5:]:
                synth_parts.append(f"  {m.sender} -> {m.recipient}: {m.content[:100]}")

        synthesis_input = "\n".join(synth_parts)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYNTHESIZER_PROMPT,
            messages=[{"role": "user", "content": synthesis_input}],
        )

        final_text = "".join(b.text for b in response.content if hasattr(b, "text"))
        print(f"\nAssistant: {final_text}")

        # Store in orchestrator memory
        self.memory.store_interaction(user_message, final_text, agents_used)

        # Mark goals as completed if applicable
        for goal in plan.get("goals", []):
            self.memory.update_goal(goal, "completed")

        return final_text

    def show_system_status(self):
        """Display the full system status: all agents, memory, goals, messages."""
        print(f"\n{'='*60}")
        print("SYSTEM STATUS")
        print(f"{'='*60}")

        print("\n[Orchestrator Memory]")
        print(self.memory.get_memory_summary())

        for agent_id, agent in self.agents.items():
            print(f"\n[{agent_id}]")
            print(f"  Status: {agent.status}")
            print(agent.memory.get_memory_summary())

        msgs = self.bus.get_all_messages()
        print(f"\n[Message Bus] Total messages: {len(msgs)}")
        for m in msgs[-5:]:
            print(f"  {m.sender} -> {m.recipient} ({m.msg_type}): {m.content[:60]}")

    def clear_all_memory(self):
        """Reset all agent memories and message bus."""
        self.memory.clear()
        for agent in self.agents.values():
            agent.memory.clear()
        self.bus.clear()
        print("All memory and messages cleared.")
