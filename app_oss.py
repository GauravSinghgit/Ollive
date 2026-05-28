"""Standalone OSS assistant app — Qwen2.5-0.5B-Instruct."""

import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from assistants.oss_assistant import OSSAssistant
from memory.conversation import ConversationMemory
from observability.logger import default_logger

SYSTEM_PROMPTS = {
    "General Assistant": (
        "You are a helpful, harmless, and honest AI assistant. "
        "Answer questions accurately and concisely."
    ),
    "Coding Helper": (
        "You are an expert software engineer. "
        "Help with code, debugging, and explanations."
    ),
    "Creative Writer": (
        "You are a creative writing assistant. "
        "Help brainstorm and write engaging content."
    ),
}

_assistant = None


def get_assistant(system_prompt: str):
    global _assistant
    if _assistant is None:
        _assistant = OSSAssistant(system_prompt=system_prompt)
    else:
        _assistant.system_prompt = system_prompt
    return _assistant


def chat(user_message: str, history, memory: ConversationMemory, system_choice: str):
    if not user_message.strip():
        return history, memory, ""
    assistant = get_assistant(SYSTEM_PROMPTS[system_choice])
    from observability.logger import timer
    with timer() as t:
        response = assistant.chat(user_message, history=memory, category="general")
    history = history + [[user_message, response]]
    return history, memory, f"⏱ {t.elapsed_ms:.0f} ms  |  turns: {len(memory) // 2}"


def clear(memory: ConversationMemory):
    memory.clear()
    return [], memory, ""


with gr.Blocks(title="OSS Assistant — Qwen2.5-0.5B") as demo:
    gr.Markdown("# 🟦 OSS Assistant\n**Qwen2.5-0.5B-Instruct** · HuggingFace Transformers")

    with gr.Row():
        system_choice = gr.Dropdown(
            choices=list(SYSTEM_PROMPTS.keys()),
            value="General Assistant",
            label="Mode",
            scale=3,
        )
        clear_btn = gr.Button("🗑 Clear", variant="secondary", scale=1)

    chatbot = gr.Chatbot(height=480)
    status = gr.Markdown("")

    with gr.Row():
        msg = gr.Textbox(placeholder="Type a message …", label="", scale=5, autofocus=True)
        send = gr.Button("Send", variant="primary", scale=1)

    memory_state = gr.State(ConversationMemory())

    inputs = [msg, chatbot, memory_state, system_choice]
    outputs = [chatbot, memory_state, status]

    send.click(chat, inputs=inputs, outputs=outputs).then(lambda: "", outputs=[msg])
    msg.submit(chat, inputs=inputs, outputs=outputs).then(lambda: "", outputs=[msg])
    clear_btn.click(clear, inputs=[memory_state], outputs=[chatbot, memory_state, status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, theme=gr.themes.Soft())
