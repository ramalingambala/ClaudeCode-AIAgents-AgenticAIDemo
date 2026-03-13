"""Research Agent — specialized in knowledge base searches with persistent memory.

Remembers past searches and their results, building up knowledge over time.
Can suggest related topics based on search history.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.search import search_knowledge_base
from ai_agents.base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    name = "research_agent"
    description = "Specialist in knowledge base research. Remembers past searches and findings."
    system_prompt = (
        "You are a research specialist agent with persistent memory. "
        "You remember past searches and what articles were found. "
        "If the user searches for something you've seen before, mention it. "
        "You can suggest related topics based on your memory. "
        "Always use the search_knowledge_base tool to find articles."
    )
    tools = [
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
    tool_functions = {"search_knowledge_base": search_knowledge_base}

    def _learn_from_result(self, tool_name, tool_input, result):
        if tool_name == "search_knowledge_base":
            query = result.get("query", "")
            total = result.get("total_results", 0)
            titles = [r["title"] for r in result.get("results", [])]
            self.memory.store_fact(
                f"Searched '{query}': found {total} results — {', '.join(titles) if titles else 'none'}",
                source="search_knowledge_base tool",
            )
            # Share findings with other agents
            if titles:
                self.share_info(
                    f"Found articles for '{query}': {', '.join(titles)}"
                )
