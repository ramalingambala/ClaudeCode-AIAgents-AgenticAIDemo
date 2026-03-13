"""Base Agent class with persistent memory, goals, and inter-agent communication.

All specialized agents inherit from this class to get:
- Persistent memory (facts + conversation history)
- Goal-directed behavior
- Inter-agent communication via message bus
- Self-monitoring and adaptive re-planning
"""

import json
import os
import time

from anthropic import Anthropic
import httpx

from ai_agents.memory.store import MemoryStore
from ai_agents.communication.message_bus import MessageBus


def create_client() -> Anthropic:
    return Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        http_client=httpx.Client(verify=False),
    )


class BaseAgent:
    """Base class for all AI Agents with memory, goals, and communication."""

    name: str = "base_agent"
    description: str = "A base agent."
    system_prompt: str = "You are a helpful assistant."
    tools: list[dict] = []
    tool_functions: dict = {}

    def __init__(self, message_bus: MessageBus):
        self.memory = MemoryStore(self.name)
        self.bus = message_bus
        self._last_msg_check = 0
        self.status = "idle"  # idle, working, waiting, error

    def _build_context_prompt(self, task: str) -> str:
        """Build a prompt enriched with memory and inter-agent messages."""
        parts = [self.system_prompt]

        # Inject relevant memories
        recent = self.memory.get_recent_interactions(limit=3)
        if recent:
            parts.append("\n## Your Recent Memory (past interactions):")
            for interaction in recent:
                parts.append(
                    f"- User asked: \"{interaction['user_query'][:100]}\" "
                    f"-> You responded with tools: {interaction['tools_used']}"
                )

        # Inject relevant facts
        facts = self.memory.recall_facts(query=task, limit=5)
        if facts:
            parts.append("\n## Known Facts:")
            for f in facts:
                parts.append(f"- {f['fact']}")

        # Inject active goals
        goals = self.memory.get_active_goals()
        if goals:
            parts.append("\n## Your Active Goals:")
            for g in goals:
                parts.append(f"- {g['goal']} (status: {g['status']})")

        # Inject messages from other agents
        messages = self.bus.get_unread_for(self.name, since=self._last_msg_check)
        if messages:
            parts.append("\n## Messages from Other Agents:")
            for m in messages:
                parts.append(f"- From {m.sender} ({m.msg_type}): {m.content}")
            self._last_msg_check = time.time()

        return "\n".join(parts)

    def run(self, task: str) -> dict:
        """Execute a task with the agent loop, memory, and communication."""
        self.status = "working"
        print(f"\n  [{self.name}] Starting task: {task}")

        client = create_client()
        context_prompt = self._build_context_prompt(task)
        messages = [{"role": "user", "content": task}]
        tools_used = []
        max_retries = 2
        attempt = 0

        while True:
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=512,
                    system=context_prompt,
                    tools=self.tools,
                    messages=messages,
                )
            except Exception as e:
                attempt += 1
                print(f"  [{self.name}] Error (attempt {attempt}): {e}")
                if attempt >= max_retries:
                    self.status = "error"
                    # Notify orchestrator about failure
                    self.bus.send(
                        self.name, "orchestrator",
                        f"Failed after {max_retries} attempts: {e}",
                        msg_type="error",
                    )
                    return {"agent": self.name, "status": "error", "result": str(e)}
                # Re-plan: adjust and retry
                print(f"  [{self.name}] Re-planning...")
                continue

            if response.stop_reason == "tool_use":
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})
                tool_results = []

                for block in assistant_content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        print(f"  [{self.name}] Tool: {tool_name}({json.dumps(block.input)})")
                        tools_used.append(tool_name)

                        try:
                            result = self.tool_functions[tool_name](**block.input)
                            print(f"  [{self.name}] Result: {json.dumps(result, indent=2)}")

                            # Learn from tool results — store as facts
                            self._learn_from_result(tool_name, block.input, result)

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result),
                            })
                        except Exception as e:
                            print(f"  [{self.name}] Tool error: {e}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps({"error": str(e)}),
                                "is_error": True,
                            })

                messages.append({"role": "user", "content": tool_results})
            else:
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                print(f"  [{self.name}] Done")

                # Store this interaction in memory
                self.memory.store_interaction(task, text, tools_used)

                self.status = "idle"
                return {"agent": self.name, "status": "success", "result": text}

    def _learn_from_result(self, tool_name: str, tool_input: dict, result: dict):
        """Extract and store facts from tool results. Override in subclasses."""
        pass

    def request_help(self, target_agent: str, question: str):
        """Ask another agent for help via the message bus."""
        self.bus.send(self.name, target_agent, question, msg_type="request")

    def share_info(self, info: str):
        """Broadcast information to all agents."""
        self.bus.send(self.name, "broadcast", info, msg_type="info")

    def handoff(self, target_agent: str, task: str, context: str):
        """Hand off a task to another agent with context."""
        self.bus.send(
            self.name, target_agent, task,
            msg_type="handoff",
            metadata={"context": context},
        )
