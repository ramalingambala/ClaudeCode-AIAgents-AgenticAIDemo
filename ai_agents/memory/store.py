"""Persistent memory store for AI Agents.

Provides both short-term (conversation) and long-term (cross-session) memory
using JSON file storage. Each agent gets its own memory namespace.
"""

import json
import os
import time
from pathlib import Path

MEMORY_DIR = Path(__file__).parent / "data"


class MemoryStore:
    """Persistent memory for an agent, stored as JSON files."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memory_file = MEMORY_DIR / f"{agent_id}.json"
        self._ensure_storage()

    def _ensure_storage(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self._save({"facts": [], "conversation_history": [], "goals": []})

    def _load(self) -> dict:
        with open(self.memory_file, "r") as f:
            return json.load(f)

    def _save(self, data: dict):
        with open(self.memory_file, "w") as f:
            json.dump(data, f, indent=2)

    # --- Facts: long-term knowledge the agent has learned ---

    def store_fact(self, fact: str, source: str = ""):
        data = self._load()
        entry = {
            "fact": fact,
            "source": source,
            "timestamp": time.time(),
        }
        # Avoid duplicate facts
        existing = [f["fact"] for f in data["facts"]]
        if fact not in existing:
            data["facts"].append(entry)
            self._save(data)

    def recall_facts(self, query: str = "", limit: int = 10) -> list[dict]:
        data = self._load()
        facts = data["facts"]
        if query:
            query_lower = query.lower()
            facts = [f for f in facts if query_lower in f["fact"].lower()]
        return facts[-limit:]

    # --- Conversation history: remembers past interactions ---

    def store_interaction(self, user_query: str, agent_response: str, tools_used: list[str]):
        data = self._load()
        entry = {
            "user_query": user_query,
            "agent_response": agent_response,
            "tools_used": tools_used,
            "timestamp": time.time(),
        }
        data["conversation_history"].append(entry)
        # Keep last 50 interactions
        data["conversation_history"] = data["conversation_history"][-50:]
        self._save(data)

    def get_recent_interactions(self, limit: int = 5) -> list[dict]:
        data = self._load()
        return data["conversation_history"][-limit:]

    # --- Goals: agent's active objectives ---

    def add_goal(self, goal: str, status: str = "active"):
        data = self._load()
        entry = {
            "goal": goal,
            "status": status,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        data["goals"].append(entry)
        self._save(data)

    def update_goal(self, goal: str, status: str):
        data = self._load()
        for g in data["goals"]:
            if g["goal"] == goal:
                g["status"] = status
                g["updated_at"] = time.time()
        self._save(data)

    def get_active_goals(self) -> list[dict]:
        data = self._load()
        return [g for g in data["goals"] if g["status"] == "active"]

    def get_all_goals(self) -> list[dict]:
        data = self._load()
        return data["goals"]

    def clear(self):
        self._save({"facts": [], "conversation_history": [], "goals": []})

    def get_memory_summary(self) -> str:
        data = self._load()
        facts = data["facts"]
        history = data["conversation_history"]
        goals = data["goals"]
        active_goals = [g for g in goals if g["status"] == "active"]

        summary = f"Memory for {self.agent_id}:\n"
        summary += f"  Facts stored: {len(facts)}\n"
        summary += f"  Past interactions: {len(history)}\n"
        summary += f"  Active goals: {len(active_goals)}\n"

        if facts:
            summary += "\n  Recent facts:\n"
            for f in facts[-3:]:
                summary += f"    - {f['fact']}\n"

        if history:
            summary += "\n  Last interaction:\n"
            last = history[-1]
            summary += f"    Q: {last['user_query'][:80]}...\n"

        return summary
