"""
Shared building blocks for the multi-agent system.

Agents communicate through a shared `Blackboard` (a common pattern in
multi-agent systems): each agent reads the trip spec and the outputs of
other agents, writes its own result, and posts messages to a trace that
the UI shows as the "agent conversation".
"""

import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Message:
    sender: str
    receiver: str
    kind: str          # "task" | "tool_call" | "result" | "feedback" | "warning" | "info"
    content: str
    ts: float = field(default_factory=time.time)


class Blackboard:
    """Thread-safe shared memory + message log."""

    def __init__(self):
        self.data: dict = {}
        self.messages: list[Message] = []
        self._lock = Lock()
        self.t0 = time.time()
        self.timings: dict[str, float] = {}

    def post(self, sender, receiver, kind, content):
        with self._lock:
            self.messages.append(Message(sender, receiver, kind, content))

    def put(self, key, value):
        with self._lock:
            self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

    def trace(self) -> list[dict]:
        return [{"t (s)": round(m.ts - self.t0, 2), "from": m.sender, "to": m.receiver,
                 "type": m.kind, "message": m.content} for m in self.messages]


class Agent:
    name = "Agent"
    role = ""
    tools: list[str] = []

    def __init__(self, board: Blackboard, llm=None):
        self.board = board
        self.llm = llm

    def say(self, receiver, kind, content):
        self.board.post(self.name, receiver, kind, content)

    def timed(self, fn, *args, **kwargs):
        t = time.time()
        out = fn(*args, **kwargs)
        self.board.timings[self.name] = self.board.timings.get(self.name, 0) + time.time() - t
        return out
