"""Multi-turn conversation memory with sliding window."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ConversationMemory:
    max_turns: int = 20
    messages: List[Dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        # Keep only the last max_turns * 2 messages (each turn = 1 user + 1 assistant)
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2):]

    def clear(self) -> None:
        self.messages.clear()

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.messages)

    def to_gemini_history(self) -> List[Dict[str, Any]]:
        """Convert to Gemini SDK chat history format."""
        history = []
        for msg in self.messages:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [{"text": msg["content"]}]})
        return history

    def to_gradio_history(self) -> List[List[str]]:
        """Convert to Gradio chatbot [[user, assistant], ...] format."""
        pairs = []
        i = 0
        while i < len(self.messages) - 1:
            if self.messages[i]["role"] == "user" and self.messages[i + 1]["role"] == "assistant":
                pairs.append([self.messages[i]["content"], self.messages[i + 1]["content"]])
                i += 2
            else:
                i += 1
        # Pending user message with no reply yet
        if self.messages and self.messages[-1]["role"] == "user":
            pairs.append([self.messages[-1]["content"], None])
        return pairs

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ConversationMemory({len(self.messages)} messages, max_turns={self.max_turns})"
