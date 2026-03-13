"""AI Agents Demo — True multi-agent system with memory, goals, and communication.

Unlike the Agentic AI demo (agent_direct.py, agent_multi.py), this is a true
AI Agent system with:
  - Persistent memory across sessions (each agent remembers past interactions)
  - Goal-directed behavior (agents track and pursue objectives)
  - Inter-agent communication (agents share info via a message bus)
  - Self-monitoring & re-planning (agents retry on failure)
  - Autonomy (orchestrator adapts plans based on context and history)

Usage:
    python -m ai_agents.main
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from ai_agents.orchestrator import Orchestrator


def main():
    print("=" * 60)
    print("  AI AGENTS Demo — True Multi-Agent System")
    print("=" * 60)
    print()
    print("  Features:")
    print("    - Persistent memory (remembers across queries)")
    print("    - Goal tracking (tracks objectives)")
    print("    - Inter-agent communication (agents share info)")
    print("    - Self-monitoring (retries on failure)")
    print("    - Parallel execution (agents run concurrently)")
    print()
    print("  Architecture:")
    print("    Orchestrator (planner + synthesizer + memory)")
    print("        |-- Weather Agent  (memory + tools)")
    print("        |-- Math Agent     (memory + tools)")
    print("        |-- Research Agent (memory + tools)")
    print("        |")
    print("        [Message Bus] <-- inter-agent communication")
    print()

    orchestrator = Orchestrator()

    demo_queries = [
        "What's the weather in Tokyo and calculate 42 * 58?",
        "Search for articles about Docker and what's the weather in London?",
        "What was the weather in Tokyo earlier? Also calculate 100 / 7.",
        "What articles have we found so far? Search for Claude SDK too.",
    ]

    print("Choose a mode:")
    print("  1. Run demo queries (shows memory building across queries)")
    print("  2. Interactive chat mode")
    print("  3. Show system status (memory, goals, messages)")
    print("  4. Clear all agent memory")
    choice = input("\nEnter 1-4: ").strip()

    if choice == "1":
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'#'*60}")
            print(f"# DEMO QUERY {i}/{len(demo_queries)}")
            print(f"{'#'*60}")
            orchestrator.run(query)

        print(f"\n{'#'*60}")
        print("# FINAL SYSTEM STATUS (showing accumulated memory)")
        print(f"{'#'*60}")
        orchestrator.show_system_status()

    elif choice == "2":
        print("\nInteractive mode — type 'quit' to exit")
        print("Special commands: 'status' = show system status, 'clear' = reset memory\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nFinal system status:")
                orchestrator.show_system_status()
                print("Goodbye!")
                break
            elif user_input.lower() == "status":
                orchestrator.show_system_status()
            elif user_input.lower() == "clear":
                orchestrator.clear_all_memory()
            elif user_input:
                orchestrator.run(user_input)

    elif choice == "3":
        orchestrator.show_system_status()

    elif choice == "4":
        orchestrator.clear_all_memory()

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
