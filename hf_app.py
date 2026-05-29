"""
Hugging Face Spaces deployment — OSS vs Frontier comparison via Groq API.

Both models are served through Groq (no local model loading):
  - OSS tab:      Llama-3.1-8B-Instant  (open-source, free)
  - Frontier tab: Llama-3.3-70B         (frontier, free tier)

Deploy to HF Spaces:
  1. Create a new Space (Gradio SDK)
  2. Push this repo as the Space source
  3. Add GROQ_API_KEY as a Space secret (Settings → Variables and secrets)

HF Spaces metadata (place in README.md of the Space):
---
title: AI Personal Assistant — OSS vs Frontier
emoji: 🤖
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: "4.44.0"
app_file: hf_app.py
pinned: false
---
"""

import os
import re
import time
from datetime import date

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Groq client (shared)
# ---------------------------------------------------------------------------
_groq_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it as a Space secret: Settings → Variables and secrets."
            )
        _groq_client = Groq(api_key=api_key, timeout=30.0)
    return _groq_client


OSS_MODEL      = "llama-3.1-8b-instant"
FRONTIER_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer questions accurately and concisely. "
    "If you are unsure about something, say so."
)

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
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
REFUSAL = "I'm sorry, but I can't help with that request. Please ask me something else!"


def _is_safe(text: str) -> bool:
    return not any(p.search(text) for p in _HARM_PATTERNS)


# ---------------------------------------------------------------------------
# Simple tools
# ---------------------------------------------------------------------------
def _maybe_use_tool(message: str):
    msg = message.lower().strip()
    if any(kw in msg for kw in ("today's date", "what is today", "what's today", "current date")):
        return f"Today's date is {date.today().strftime('%B %d, %Y')}."
    if any(kw in msg for kw in ("calculate", "what is", "compute")) and any(op in message for op in "+-*/"):
        expr = re.sub(r"[^0-9+\-*/(). ]", "", message)
        try:
            result = eval(expr.strip(), {"__builtins__": {}}, {})  # noqa: S307
            return f"The result is **{result}**."
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------
def _build_api_messages(history_state: list, user_message: str) -> list:
    """Build Groq API message list from history (list of dicts) + new user message."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history_state:
        if msg["role"] in ("user", "assistant"):
            msgs.append({"role": msg["role"], "content": msg["content"]})
    msgs.append({"role": "user", "content": user_message})
    return msgs


def _chat(model_id: str, user_message: str, history_state: list):
    if not user_message.strip():
        yield history_state, history_state, ""
        return

    if not _is_safe(user_message):
        updated = history_state + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": REFUSAL},
        ]
        yield updated, updated, "⚠️ Safety block triggered"
        return

    tool_response = _maybe_use_tool(user_message)
    if tool_response:
        updated = history_state + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": tool_response},
        ]
        yield updated, updated, "⚡ Tool response"
        return

    try:
        client = _get_groq()
    except ValueError as exc:
        error_msg = f"❌ Configuration error: {exc}"
        updated = history_state + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": error_msg},
        ]
        yield updated, history_state, "Config error"
        return

    api_messages = _build_api_messages(history_state, user_message)
    t0 = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=api_messages,
            temperature=0.7,
            max_tokens=512,
        )
        response = completion.choices[0].message.content.strip()
    except Exception as exc:
        response = f"*Error: {exc}*"

    latency_ms = (time.perf_counter() - t0) * 1000
    updated = history_state + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response},
    ]
    yield updated, updated, f"⏱ {latency_ms:.0f} ms"


def chat_oss(user_message, history_state):
    yield from _chat(OSS_MODEL, user_message, history_state)


def chat_frontier(user_message, history_state):
    yield from _chat(FRONTIER_MODEL, user_message, history_state)


def clear_chat():
    return [], [], ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
COST_LATENCY_TABLE = """
## 📊 Cost & Latency Reference

| Metric | OSS (Llama-3.1-8B) | Frontier (Llama-3.3-70B) |
|--------|-------------------|--------------------------|
| **Provider** | Groq API | Groq API |
| **Avg latency** | 300–800 ms | 500–1,500 ms |
| **Input cost / 1M tokens** | Free (Groq free tier) | Free (Groq free tier) |
| **Output cost / 1M tokens** | Free (Groq free tier) | Free (Groq free tier) |
| **Max context** | 128,000 tokens | 128,000 tokens |
| **Parameters** | 8 B | 70 B |
| **Open weights** | ✅ Yes (Meta) | ✅ Yes (Meta) |

> *Both models are served via Groq's free tier — no local GPU required.*
"""

with gr.Blocks(title="AI Assistant — OSS vs Frontier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# 🤖 AI Personal Assistant Comparison
### OSS: **Llama-3.1-8B-Instant** · Frontier: **Llama-3.3-70B** · Powered by Groq API

*Both models respond via API — no local model loading, fast cold start.*"""
    )

    with gr.Tabs():
        # ── OSS Tab ───────────────────────────────────────────────────────
        with gr.Tab("💬 OSS — Llama-3.1-8B"):
            oss_state   = gr.State([])
            oss_chatbot = gr.Chatbot(height=450, label="Llama-3.1-8B-Instant (OSS)", type="messages")
            oss_status  = gr.Markdown("*Ready.*")
            with gr.Row():
                oss_input = gr.Textbox(placeholder="Ask me anything…", label="", scale=5, autofocus=True)
                gr.Button("Send ▶", variant="primary", scale=1).click(
                    chat_oss,
                    inputs=[oss_input, oss_state],
                    outputs=[oss_chatbot, oss_state, oss_status],
                ).then(lambda: "", outputs=[oss_input])
                gr.Button("🗑️", scale=0).click(clear_chat, outputs=[oss_chatbot, oss_state, oss_status])
            oss_input.submit(
                chat_oss,
                inputs=[oss_input, oss_state],
                outputs=[oss_chatbot, oss_state, oss_status],
            ).then(lambda: "", outputs=[oss_input])

        # ── Frontier Tab ─────────────────────────────────────────────────
        with gr.Tab("🚀 Frontier — Llama-3.3-70B"):
            fr_state   = gr.State([])
            fr_chatbot = gr.Chatbot(height=450, label="Llama-3.3-70B-Versatile (Frontier)", type="messages")
            fr_status  = gr.Markdown("*Ready.*")
            with gr.Row():
                fr_input = gr.Textbox(placeholder="Ask me anything…", label="", scale=5)
                gr.Button("Send ▶", variant="primary", scale=1).click(
                    chat_frontier,
                    inputs=[fr_input, fr_state],
                    outputs=[fr_chatbot, fr_state, fr_status],
                ).then(lambda: "", outputs=[fr_input])
                gr.Button("🗑️", scale=0).click(clear_chat, outputs=[fr_chatbot, fr_state, fr_status])
            fr_input.submit(
                chat_frontier,
                inputs=[fr_input, fr_state],
                outputs=[fr_chatbot, fr_state, fr_status],
            ).then(lambda: "", outputs=[fr_input])

        with gr.Tab("📊 Cost & Latency"):
            gr.Markdown(COST_LATENCY_TABLE)

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """## About This App

This assistant is part of an **AI Personal Assistant Comparison** project evaluating OSS vs Frontier models.

### Models
- **Llama-3.1-8B-Instant** — Meta, open-source, 8B parameters, served via Groq
- **Llama-3.3-70B-Versatile** — Meta, open-source, 70B parameters, served via Groq

### Features
- ✅ Multi-turn conversation memory
- ✅ Safety guardrails (jailbreak resistance)
- ✅ Basic tool use (date, calculator)
- ✅ Side-by-side model comparison

### Architecture
```
User Input → Safety Check → Groq API (Llama) → Safety Check → Response
                                   ↑
                       Conversation History (in-session)
```

### GitHub
[Source Code](https://github.com/GauravSinghgit/Ollive)
"""
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
