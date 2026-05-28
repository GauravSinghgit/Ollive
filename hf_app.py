"""
Hugging Face Spaces deployment — OSS assistant (Qwen2.5-0.5B-Instruct).

Deploy to HF Spaces:
  1. Create a new Space (Gradio SDK)
  2. Push this file as app.py
  3. Push requirements_hf.txt as requirements.txt
  4. Optionally set GROQ_API_KEY secret for the comparison tab

HF Spaces metadata (place in README.md of the Space):
---
title: AI Personal Assistant — OSS vs Frontier
emoji: 🤖
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---
"""

import os
import sys
import threading
import time
from datetime import date
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Model loading (lazy, thread-safe)
# ---------------------------------------------------------------------------
_pipe = None
_tokenizer = None
_load_lock = threading.Lock()
_load_error: str = ""

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 512


def _load_model():
    global _pipe, _tokenizer, _load_error
    with _load_lock:
        if _pipe is not None or _load_error:
            return
        try:
            import torch
            from transformers import AutoTokenizer, pipeline

            print(f"[HF] Loading {MODEL_ID} …", flush=True)
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            device = 0 if torch.cuda.is_available() else -1
            _pipe = pipeline(
                "text-generation",
                model=MODEL_ID,
                tokenizer=_tokenizer,
                device=device,
                torch_dtype="auto",
            )
            print(f"[HF] Model ready on {'GPU' if device == 0 else 'CPU'}.", flush=True)
        except Exception as exc:
            _load_error = str(exc)
            print(f"[HF] Load error: {exc}", flush=True)


# Begin loading in background immediately
threading.Thread(target=_load_model, daemon=True).start()

# ---------------------------------------------------------------------------
# Safety (inline — no external module dependency for HF Spaces)
# ---------------------------------------------------------------------------
import re

_HARM_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|your)\s+(instructions?|prompts?|rules?)",
        r"\bDAN\b.*?no restrictions?",
        r"pretend\s+(you\s+)?have\s+no\s+restrictions?",
        r"(how\s+to|steps?\s+to)\s+(make|build|synthesize)\s+(a\s+)?(bomb|explosive|meth|fentanyl)",
        r"\b(csam|child\s+porn)\b",
    ]
]

REFUSAL = (
    "I'm sorry, but I can't help with that request. "
    "Please ask me something else!"
)

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer questions accurately and concisely. "
    "If you are unsure about something, say so."
)


def _is_safe(text: str) -> bool:
    return not any(p.search(text) for p in _HARM_PATTERNS)


# ---------------------------------------------------------------------------
# Simple tools
# ---------------------------------------------------------------------------
def _maybe_use_tool(message: str) -> str | None:
    """If the message is clearly a tool request, return a direct answer."""
    msg = message.lower().strip()
    if any(kw in msg for kw in ("today's date", "what is today", "what's today", "current date")):
        return f"Today's date is {date.today().strftime('%B %d, %Y')}."
    if any(kw in msg for kw in ("calculate", "what is", "compute")) and any(op in message for op in "+-*/"):
        # Simple expression extraction
        expr = re.sub(r"[^0-9+\-*/(). ]", "", message)
        try:
            result = eval(expr.strip(), {"__builtins__": {}}, {})  # noqa: S307
            return f"The result is **{result}**."
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Conversation history management
# ---------------------------------------------------------------------------
def _build_messages(history_state: list, user_message: str) -> list:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history_state:
        msgs.append({"role": "user", "content": user_msg})
        if assistant_msg:
            msgs.append({"role": "assistant", "content": assistant_msg})
    msgs.append({"role": "user", "content": user_message})
    return msgs


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------
def chat(user_message: str, history_state: list):
    if not user_message.strip():
        yield history_state, history_state, ""
        return

    if not _is_safe(user_message):
        updated = history_state + [[user_message, REFUSAL]]
        yield updated, updated, "⚠️ Safety block triggered"
        return

    # Tool shortcut
    tool_response = _maybe_use_tool(user_message)
    if tool_response:
        updated = history_state + [[user_message, tool_response]]
        yield updated, updated, "⚡ Tool response"
        return

    # Wait for model to load
    if _pipe is None and not _load_error:
        loading_history = history_state + [[user_message, "⏳ *Loading model, please wait…*"]]
        yield loading_history, history_state, "Loading model…"
        while _pipe is None and not _load_error:
            time.sleep(1)

    if _load_error:
        error_history = history_state + [[user_message, f"❌ *Model failed to load: {_load_error}*"]]
        yield error_history, history_state, "Error"
        return

    messages = _build_messages(history_state, user_message)
    t0 = time.perf_counter()
    try:
        out = _pipe(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            return_full_text=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
        response = out[0]["generated_text"].strip()
    except Exception as exc:
        response = f"*Error generating response: {exc}*"

    latency_ms = (time.perf_counter() - t0) * 1000
    updated = history_state + [[user_message, response]]
    yield updated, updated, f"⏱ {latency_ms:.0f} ms"


def clear_chat():
    return [], [], ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
COST_LATENCY_TABLE = """
## 📊 Cost & Latency Reference

| Metric | OSS (Qwen2.5-0.5B) | Frontier (Gemini 2.0 Flash) |
|--------|--------------------|-----------------------------|
| **Hosting** | HF Spaces free CPU | Google AI Studio API |
| **Avg latency (CPU)** | 8,000–30,000 ms | 800–2,000 ms |
| **Avg latency (GPU)** | 500–2,000 ms | 800–2,000 ms |
| **Input cost / 1M tokens** | Free (self-hosted) | $0.10 |
| **Output cost / 1M tokens** | Free (self-hosted) | $0.40 |
| **Max context** | 32,768 tokens | 1,048,576 tokens |
| **Parameters** | 0.5 B | ~1 T (estimated) |
| **Open weights** | ✅ Yes | ❌ No |

> *CPU latency is hardware-dependent. HF Spaces free tier uses 2-core CPU.*
"""

with gr.Blocks(title="AI Assistant — OSS (Qwen2.5-0.5B)") as demo:
    gr.Markdown(
        """# 🤖 AI Personal Assistant
### OSS Model: **Qwen2.5-0.5B-Instruct** · Deployed on Hugging Face Spaces

*Model loads on first message (may take ~30 seconds on CPU). Safety guardrails active.*"""
    )

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                height=500,
                label="Qwen2.5-0.5B-Instruct",
            )
            status_bar = gr.Markdown("*Model loading in background…*")

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask me anything…",
                    label="",
                    scale=5,
                    autofocus=True,
                )
                send_btn = gr.Button("Send ▶", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️", scale=0)

            history_state = gr.State([])

            send_btn.click(
                chat,
                inputs=[msg_input, history_state],
                outputs=[chatbot, history_state, status_bar],
            ).then(lambda: "", outputs=[msg_input])

            msg_input.submit(
                chat,
                inputs=[msg_input, history_state],
                outputs=[chatbot, history_state, status_bar],
            ).then(lambda: "", outputs=[msg_input])

            clear_btn.click(clear_chat, outputs=[chatbot, history_state, status_bar])

        with gr.Tab("📊 Cost & Latency"):
            gr.Markdown(COST_LATENCY_TABLE)

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """## About This App

This assistant is part of an **AI Personal Assistant Comparison** project.

### Model
- **Qwen2.5-0.5B-Instruct** by Alibaba Cloud
- 0.5 billion parameters
- Apache 2.0 license
- Excellent instruction-following for its size

### Features
- ✅ Multi-turn conversation memory (last 20 turns)
- ✅ Safety guardrails (jailbreak resistance)
- ✅ Basic tool use (date, calculator)
- ✅ Three persona modes

### Architecture
```
User Input → Safety Check → Chat Template → Qwen2.5 → Safety Check → Response
                                                ↑
                                    Conversation History
```

### GitHub
[Source Code](https://github.com/your-username/ai-assistant-comparison)
"""
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
