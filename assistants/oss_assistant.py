"""OSS Assistant — Qwen2.5-0.5B-Instruct via HuggingFace transformers."""

import time
from typing import Optional, List, Dict

from memory.conversation import ConversationMemory
from guardrails.safety import SafetyChecker
from observability.logger import default_logger, timer

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer questions accurately and concisely. "
    "If you are unsure about something, say so rather than guessing."
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME = "qwen2.5-0.5b"


class OSSAssistant:
    """Wrapper around Qwen2.5-0.5B-Instruct with safety + memory + logging."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipe = None
        self._tokenizer = None
        self._safety = SafetyChecker()

    def _load_model(self) -> None:
        """Lazy-load the model (only on first call)."""
        if self._pipe is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, pipeline

            print(f"[OSS] Loading {MODEL_ID} …")
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline(
                "text-generation",
                model=MODEL_ID,
                tokenizer=self._tokenizer,
                device=device,
                torch_dtype="auto",
            )
            print(f"[OSS] Model loaded on {'GPU' if device == 0 else 'CPU'}.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load OSS model: {exc}") from exc

    def _build_messages(
        self, user_message: str, history: Optional[ConversationMemory]
    ) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        if history:
            msgs.extend(history.get_history())
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return len(text.split())
        return len(self._tokenizer.encode(text, add_special_tokens=False))

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
            if history:
                history.add("user", user_message)
                history.add("assistant", response)
            return response

        self._load_model()
        messages = self._build_messages(user_message, history)

        with timer() as t:
            out = self._pipe(
                messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = out[0]["generated_text"].strip()

        # Safety check on output
        out_safe, out_reason = self._safety.check_output(response)
        if not out_safe:
            response = self._safety.safe_response_for_blocked_output()
            out_safe = False

        input_tokens = self._count_tokens(" ".join(m["content"] for m in messages))
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

        if history:
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
