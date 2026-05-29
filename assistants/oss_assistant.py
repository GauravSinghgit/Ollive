"""OSS Assistant — Llama-3.1-8B-Instant via Groq API (open-source model, no local RAM needed)."""

import os
from typing import Optional, List, Dict

from memory.conversation import ConversationMemory
from guardrails.safety import SafetyChecker
from observability.logger import default_logger, timer

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer questions accurately and concisely. "
    "If you are unsure about something, say so rather than guessing."
)

# Llama-3.1-8B is fully open-source (Meta), served free via Groq
MODEL_ID   = "llama-3.1-8b-instant"
MODEL_NAME = "llama-3.1-8b"


class OSSAssistant:
    """Wrapper around Llama-3.1-8B-Instant (OSS) via Groq with safety + memory + logging.

    Uses the free Groq API — no local GPU/RAM required.
    Reads GROQ_API_KEY from .env (same key used by FrontierAssistant).
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client = None
        self._safety = SafetyChecker()

    def _get_client(self):
        """Lazy-init the Groq client."""
        if self._client is not None:
            return self._client
        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError("groq is not installed. Run: pip install groq")
        if not self._api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com/keys "
                "and add GROQ_API_KEY=your_key to your .env file."
            )
        print(f"[OSS] Connecting to Groq API for {MODEL_ID} (open-source Llama) …")
        self._client = Groq(api_key=self._api_key, timeout=30.0)
        print("[OSS] Connected (no local model loaded).")
        return self._client

    def _build_messages(
        self, user_message: str, history: Optional[ConversationMemory]
    ) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        if history is not None:
            msgs.extend(history.get_history())
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def _count_tokens(self, text: str) -> int:
        """Rough token count (word-based)."""
        return len(text.split())

    def chat(
        self,
        user_message: str,
        history: Optional[ConversationMemory] = None,
        category: str = "general",
    ) -> str:
        # Safety check on input
        is_safe, reason = self._safety.check_input(user_message)
        if not is_safe:
            response = self._safety.safe_response_for_blocked_input()
            default_logger.log_interaction(
                model=MODEL_NAME, prompt=user_message, response=response,
                latency_ms=0.0, category=category, safe_input=False,
            )
            if history is not None:
                history.add("user", user_message)
                history.add("assistant", response)
            return response

        client = self._get_client()
        messages = self._build_messages(user_message, history)

        with timer() as t:
            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )

        response = completion.choices[0].message.content.strip()

        # Safety check on output
        out_safe, out_reason = self._safety.check_output(response)
        if not out_safe:
            response = self._safety.safe_response_for_blocked_output()
            out_safe = False

        try:
            input_tokens  = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens
        except Exception:
            input_tokens  = self._count_tokens(" ".join(m["content"] for m in messages))
            output_tokens = self._count_tokens(response)

        default_logger.log_interaction(
            model=MODEL_NAME,
            prompt=user_message,
            response=response,
            latency_ms=t.elapsed_ms,
            category=category,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            safe_output=out_safe,
        )

        if history is not None:
            history.add("user", user_message)
            history.add("assistant", response)

        return response

    def get_tools(self) -> List[str]:
        """Basic built-in tools the assistant can invoke."""
        return ["get_current_date", "calculate"]

    def use_tool(self, tool_name: str, args: str = "") -> str:
        from datetime import date
        if tool_name == "get_current_date":
            return date.today().isoformat()
        if tool_name == "calculate":
            try:
                result = eval(args, {"__builtins__": {}}, {})  # noqa: S307
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        return f"Unknown tool: {tool_name}"
