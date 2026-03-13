"""Inter-agent communication via a shared message bus.

Agents can send messages to each other, request help from other agents,
and broadcast information to all agents.
"""

import time
import threading
from dataclasses import dataclass, field


@dataclass
class Message:
    sender: str
    recipient: str  # agent_id or "broadcast"
    content: str
    msg_type: str  # "request", "response", "info", "handoff"
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class MessageBus:
    """Thread-safe message bus for inter-agent communication."""

    def __init__(self):
        self._messages: list[Message] = []
        self._lock = threading.Lock()
        self._handlers: dict[str, callable] = {}

    def send(self, sender: str, recipient: str, content: str,
             msg_type: str = "info", metadata: dict = None) -> Message:
        msg = Message(
            sender=sender,
            recipient=recipient,
            content=content,
            msg_type=msg_type,
            metadata=metadata or {},
        )
        with self._lock:
            self._messages.append(msg)
            print(f"  [MessageBus] {sender} -> {recipient} ({msg_type}): {content[:80]}")
        return msg

    def get_messages_for(self, agent_id: str) -> list[Message]:
        with self._lock:
            return [
                m for m in self._messages
                if m.recipient == agent_id or m.recipient == "broadcast"
            ]

    def get_unread_for(self, agent_id: str, since: float = 0) -> list[Message]:
        with self._lock:
            return [
                m for m in self._messages
                if (m.recipient == agent_id or m.recipient == "broadcast")
                and m.timestamp > since
                and m.sender != agent_id
            ]

    def get_conversation(self, agent_a: str, agent_b: str) -> list[Message]:
        with self._lock:
            return [
                m for m in self._messages
                if (m.sender == agent_a and m.recipient == agent_b)
                or (m.sender == agent_b and m.recipient == agent_a)
            ]

    def get_all_messages(self) -> list[Message]:
        with self._lock:
            return list(self._messages)

    def clear(self):
        with self._lock:
            self._messages.clear()
