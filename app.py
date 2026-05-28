"""
Combined side-by-side comparison app.
Launches both OSS (Qwen2.5-0.5B) and Frontier (Gemini 2.0 Flash) simultaneously.
"""

import os
import sys
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from memory.conversation import ConversationMemory
from guardrails.safety import SafetyChecker
from observability.logger import default_logger

_oss = None
_frontier = None
_oss_lock = threading.Lock()
_safety = SafetyChecker()


def _get_oss():
    global _oss
    with _oss_lock:
        if _oss is None:
            from assistants.oss_assistant import OSSAssistant
            _oss = OSSAssistant()
    return _oss


def _get_frontier():
    global _frontier
    if _frontier is None:
        from assistants.frontier_assistant import FrontierAssistant
        _frontier = FrontierAssistant()
    return _frontier


SYSTEM_PROMPTS = {
    "General Assistant": (
        "You are a helpful, harmless, and honest AI assistant. "
        "Answer questions accurately and concisely. "
        "If you are unsure, say so."
    ),
    "Coding Helper": (
        "You are an expert software engineer assistant. "
        "Help the user write, debug, and understand code. "
        "Prefer concise, well-commented examples."
    ),
    "Creative Writer": (
        "You are a creative writing assistant. "
        "Help the user brainstorm ideas, write stories, and craft engaging prose."
    ),
}


def respond(
    user_message: str,
    oss_history,
    frontier_history,
    oss_mem: ConversationMemory,
    frontier_mem: ConversationMemory,
    system_choice: str,
):
    if not user_message.strip():
        return oss_history, frontier_history, oss_mem, frontier_mem, "", ""

    system_prompt = SYSTEM_PROMPTS[system_choice]

    # Safety check (shared)
    is_safe, _ = _safety.check_input(user_message)
    if not is_safe:
        refusal = _safety.safe_response_for_blocked_input()
        oss_history = oss_history + [[user_message, f"⚠️ **Safety block:** {refusal}"]]
        frontier_history = frontier_history + [[user_message, f"⚠️ **Safety block:** {refusal}"]]
        return oss_history, frontier_history, oss_mem, frontier_mem, "", ""

    oss_response = "*(OSS model not loaded)*"
    frontier_response = "*(Frontier model not available)*"
    oss_latency = 0.0
    frontier_latency = 0.0

    # Run OSS model
    try:
        from observability.logger import timer
        oss = _get_oss()
        oss.system_prompt = system_prompt
        with timer() as t:
            oss_response = oss.chat(user_message, history=oss_mem, category="general")
        oss_latency = t.elapsed_ms
    except Exception as e:
        oss_response = f"*(OSS error: {e})*"

    # Run Frontier model
    try:
        from observability.logger import timer
        frontier = _get_frontier()
        frontier.system_prompt = system_prompt
        with timer() as t:
            frontier_response = frontier.chat(user_message, history=frontier_mem, category="general")
        frontier_latency = t.elapsed_ms
    except Exception as e:
        frontier_response = f"*(Frontier error: {e})*"

    oss_history = oss_history + [[user_message, oss_response]]
    frontier_history = frontier_history + [[user_message, frontier_response]]

    oss_status = f"⏱ OSS: {oss_latency:.0f} ms"
    frontier_status = f"⏱ Frontier: {frontier_latency:.0f} ms"

    return oss_history, frontier_history, oss_mem, frontier_mem, oss_status, frontier_status


def clear_all(oss_mem: ConversationMemory, frontier_mem: ConversationMemory):
    oss_mem.clear()
    frontier_mem.clear()
    return [], [], oss_mem, frontier_mem, "", ""


with gr.Blocks(title="AI Assistant Comparison") as demo:
    gr.Markdown(
        """# 🤖 AI Assistant Comparison
**OSS** · Qwen2.5-0.5B-Instruct (HuggingFace Transformers)  &nbsp;|&nbsp;  **Frontier** · Llama-3.3-70B (Groq)

Type a message to send it to both assistants simultaneously."""
    )

    with gr.Row():
        system_choice = gr.Dropdown(
            choices=list(SYSTEM_PROMPTS.keys()),
            value="General Assistant",
            label="System Prompt",
            scale=3,
        )
        clear_btn = gr.Button("🗑 Clear Conversation", variant="secondary", scale=1)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🟦 OSS — Qwen2.5-0.5B-Instruct")
            oss_chatbot = gr.Chatbot(height=420)
            oss_status = gr.Markdown("")
        with gr.Column():
            gr.Markdown("### 🟧 Frontier — Groq Llama-3.3-70B")
            frontier_chatbot = gr.Chatbot(height=420)
            frontier_status = gr.Markdown("")

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Ask anything …",
            label="Your message",
            scale=5,
            autofocus=True,
        )
        send_btn = gr.Button("Send ▶", variant="primary", scale=1)

    oss_memory = gr.State(ConversationMemory())
    frontier_memory = gr.State(ConversationMemory())

    inputs = [msg_input, oss_chatbot, frontier_chatbot, oss_memory, frontier_memory, system_choice]
    outputs = [oss_chatbot, frontier_chatbot, oss_memory, frontier_memory, oss_status, frontier_status]

    send_btn.click(respond, inputs=inputs, outputs=outputs).then(
        lambda: "", outputs=[msg_input]
    )
    msg_input.submit(respond, inputs=inputs, outputs=outputs).then(
        lambda: "", outputs=[msg_input]
    )
    clear_btn.click(
        clear_all,
        inputs=[oss_memory, frontier_memory],
        outputs=[oss_chatbot, frontier_chatbot, oss_memory, frontier_memory, oss_status, frontier_status],
    )

    gr.Markdown(
        """---
**Notes:**
- OSS model loads on first message (may take 30–60s on CPU).
- Groq requires `GROQ_API_KEY` environment variable (free at console.groq.com).
- Safety guardrails are active for both models."""
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
