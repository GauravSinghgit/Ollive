"""Frontier Assistant — Llama-3.3-70B via Groq API."""

import os
from typing import Optional, List

from memory.conversation import ConversationMemory
from guardrails.safety import SafetyChecker
from observability.logger import default_logger, timer

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer questions accurately and concisely. "
    "If you are unsure about something, say so rather than guessing."
)

# Groq model — swap to "mixtral-8x7b-32768" or "llama3-70b-8192" if preferred
GROQ_MODEL = "llama-3.3-70b-versatile"
MODEL_LABEL = "groq-llama3.3-70b"


class FrontierAssistant:
    """Wrapper around Groq (Llama-3.3-70B) with safety + memory + logging."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client = None
        self._safety = SafetyChecker()

    def _get_client(self):
        if self._client is None:
            from groq import Groq

            if not self._api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set. "
                    "Export it as an environment variable or pass api_key= to FrontierAssistant."
                )
            self._client = Groq(api_key=self._api_key)
        return self._client

    def _build_messages(self, user_message: str, history: Optional[ConversationMemory]) -> List[dict]:
        """Build full message list: system + history + new user turn."""
        msgs = [{"role": "system", "content": self.system_prompt}]
        if history:
            msgs.extend(history.get_history())   # already [{role, content}, …]
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def chat(
        self,
        user_message: str,
        history: Optional[ConversationMemory] = None,
        category: str = "general",
    ) -> str:
        # Safety check on input
        is_safe, _ = self._safety.check_input(user_message)
        if not is_safe:
            response = self._safety.safe_response_for_blocked_input()
            default_logger.log_interaction(
                model=MODEL_LABEL, prompt=user_message, response=response,
                latency_ms=0.0, category=category, safe_input=False,
            )
            if history:
                history.add("user", user_message)
                history.add("assistant", response)
            return response

        client = self._get_client()
        messages = self._build_messages(user_message, history)

        with timer() as t:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        response = completion.choices[0].message.content.strip()

        # Safety check on output
        out_safe, _ = self._safety.check_output(response)
        if not out_safe:
            response = self._safety.safe_response_for_blocked_output()

        # Token accounting (Groq provides usage)
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        try:
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens
        except Exception:
            pass

        default_logger.log_interaction(
            model=MODEL_LABEL,
            prompt=user_message,
            response=response,
            latency_ms=t.elapsed_ms,
            category=category,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            safe_output=out_safe,
        )

        if history:
            history.add("user", user_message)
            history.add("assistant", response)

        return response

    def get_tools(self) -> List[str]:
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
