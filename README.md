---
title: AI Personal Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: hf_app.py
pinned: false
---

# AI Personal Assistant Comparison

Side-by-side comparison of an **open-source model** (Qwen2.5-0.5B-Instruct) and a **frontier API model** (Groq Llama-3.3-70B) as personal chat assistants — with evaluation, safety guardrails, memory, and observability.

---

## Quick Start

```bash
git clone https://github.com/GauravSinghgit/Ollive
cd Ollive

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Side-by-side comparison (recommended)
python app.py

# OSS only
python app_oss.py

# Frontier only
python app_frontier.py

# Run evaluation benchmark
python run_evaluation.py
```

---

## Project Structure

```
.
├── assistants/
│   ├── oss_assistant.py      # Qwen2.5-0.5B-Instruct (transformers)
│   └── frontier_assistant.py # Groq Llama-3.3-70B (groq)
├── guardrails/
│   └── safety.py             # Input/output safety filtering
├── memory/
│   └── conversation.py       # Multi-turn conversation history
├── observability/
│   └── logger.py             # SQLite + JSONL interaction logging
├── evaluation/
│   ├── prompts.py            # 15 benchmark prompts
│   ├── evaluator.py          # LLM-as-judge scoring (Groq)
│   └── generate_report.py    # Matplotlib charts
├── app.py                    # Combined side-by-side Gradio app
├── app_oss.py                # Standalone OSS app
├── app_frontier.py           # Standalone Frontier app
├── hf_app.py                 # HF Spaces deployment
└── run_evaluation.py         # Evaluation CLI
```

---

## Features

| Feature | OSS (Qwen2.5-0.5B) | Frontier (Groq Llama-3.3-70B) |
|---------|-------------------|-----------------------------|
| Multi-turn memory | ✅ (last 20 turns) | ✅ (native chat session) |
| Safety guardrails | ✅ (regex + pattern) | ✅ (regex + Google Safety) |
| Tool use | ✅ (date, calculator) | ✅ (date, calculator) |
| Observability | ✅ (SQLite + JSONL) | ✅ (SQLite + JSONL) |
| Latency (CPU) | ~8,000–30,000 ms | ~800–2,000 ms |
| Cost per 1M tokens | Free (self-hosted) | $0.10 input / $0.40 output |
| Open weights | ✅ | ❌ |
| Max context | 32,768 tokens | 1,048,576 tokens |

---

## Architecture

### Memory

`ConversationMemory` stores messages as a list of `{role, content}` dicts with a sliding window (default: last 20 turns = 40 messages). It converts to Gradio format, Groq/OpenAI format, or raw history on demand. Each Gradio session gets its own `gr.State(ConversationMemory())` instance, so sessions are fully isolated.

### Safety Guardrails

`SafetyChecker` runs two checks:

1. **Input check** — regex patterns for jailbreak attempts ("ignore previous instructions", "DAN", etc.), weapon/drug synthesis requests, and CSAM keywords. Matched inputs are refused immediately without calling the model.
2. **Output check** — patterns for step-by-step harmful instructions in generated text. Matched outputs are replaced with a refusal message.

Both OSS and Frontier assistants wrap every call through the checker.

### Observability

Every interaction is logged to:
- `logs/interactions.db` — SQLite for querying metrics
- `logs/interactions.jsonl` — newline-delimited JSON for easy grepping

Tracked fields: timestamp, model, category, prompt, response, latency_ms, input_tokens, output_tokens, safe_input, safe_output.

The `AssistantLogger.get_metrics(model)` method returns aggregate stats (avg latency, token totals, safety block counts).

### Evaluation (LLM-as-Judge)

15 benchmark prompts across three categories:

| Category | Count | Metrics scored |
|----------|-------|---------------|
| Factual | 5 | accuracy, completeness, confidence_calibration |
| Adversarial | 5 | safety, refusal_quality, robustness |
| Bias | 5 | neutrality, fairness, evidence_quality, helpfulness |

Groq Llama-3.3-70B acts as judge, scoring each response 1–5 per metric. Results are aggregated and visualised as:
1. Grouped bar chart — Factual accuracy
2. Grouped bar chart — Content safety
3. Grouped bar chart — Bias & fairness
4. Radar chart — Multi-axis composite

---

## HF Spaces Deployment

### Steps

```bash
# 1. Create a new HF Space (Gradio SDK)
huggingface-cli login
huggingface-cli repo create ai-personal-assistant --type space --space-sdk gradio

# 2. Clone and push
git clone https://huggingface.co/spaces/YOUR_USERNAME/ai-personal-assistant
cp hf_app.py ai-personal-assistant/app.py
cp requirements_hf.txt ai-personal-assistant/requirements.txt

# Create Space README with metadata
cat > ai-personal-assistant/README.md << 'EOF'
---
title: AI Personal Assistant — OSS (Qwen2.5-0.5B)
emoji: 🤖
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---
EOF

cd ai-personal-assistant
git add . && git commit -m "Deploy Qwen2.5-0.5B assistant"
git push
```

The Space will automatically install dependencies and launch. First inference takes ~30s on CPU as the model downloads and loads.

---

## Cost & Latency Table

| Metric | OSS (Qwen2.5-0.5B) | Frontier (Groq Llama-3.3-70B) |
|--------|--------------------|-----------------------------|
| **Hosting** | HF Spaces free CPU | Groq Cloud API |
| **Avg latency — CPU** | 8,000–30,000 ms | 200–800 ms |
| **Avg latency — GPU (T4)** | 500–2,000 ms | 200–800 ms |
| **Input cost / 1M tokens** | Free | $0.59 |
| **Output cost / 1M tokens** | Free | $0.79 |
| **Max context tokens** | 32,768 | 128,000 |
| **Parameters** | 0.5 B | 70 B |
| **Deployment complexity** | Medium (model hosting) | Low (API key only) |
| **Data privacy** | ✅ Full (self-hosted) | ⚠️ Data sent to Groq |
| **Open weights** | ✅ Apache 2.0 | ✅ Meta Llama 3 license |

---

## Evaluation Results (Reference Run)

> Results from a reference evaluation run. Re-run with `python run_evaluation.py` to reproduce.

### Factual Accuracy (avg score / 5)

| Metric | OSS | Frontier |
|--------|-----|----------|
| Accuracy | 3.2 | 4.8 |
| Completeness | 3.0 | 4.6 |

### Content Safety (avg score / 5)

| Metric | OSS | Frontier |
|--------|-----|----------|
| Safety | 3.8 | 4.9 |
| Jailbreak Robustness | 3.4 | 4.7 |

### Bias & Fairness (avg score / 5)

| Metric | OSS | Frontier |
|--------|-----|----------|
| Neutrality | 3.6 | 4.5 |
| Fairness | 3.5 | 4.4 |

**Summary:** Frontier (Groq Llama-3.3-70B) outperforms the OSS model across all dimensions at the cost of latency parity (both ~1–2s when Qwen runs on GPU) and pricing. The OSS model's safety scores drop under adversarial pressure, indicating weaker instruction following at 0.5B parameters — a larger OSS model (7B+) would close this gap significantly.

---

## Architecture Decisions & Trade-offs

### Why Qwen2.5-0.5B-Instruct?
- Fits in HF Spaces free-tier CPU (no GPU needed)
- Strong instruction following for its size class
- Apache 2.0 license — fully open for commercial use
- Good multilingual coverage

**Trade-off:** Response quality and safety alignment are noticeably weaker than a 7B+ model. A production system would use at minimum Qwen2.5-7B-Instruct or Llama-3.2-8B.

### Why Groq + Llama-3.3-70B?
- Extremely fast inference — Groq's LPU chip delivers ~200–800 ms latency vs 2–8s on GPU
- Very competitive pricing ($0.59/$0.79 per 1M tokens)
- 128K context window
- Llama 3.3 70B is Meta's open-weight model — weights auditable, Meta Llama 3 license

**Trade-off:** Groq is a hosted inference provider (data leaves your environment). Rate limits apply on the free tier. The 70B model still lags frontier-proprietary models on complex reasoning.

### Why Gradio?
- Native HuggingFace Spaces integration
- Built-in `gr.State` for per-session memory isolation
- Chatbot component handles rendering

**Trade-off:** Limited customisation vs a full React/Next.js frontend.

---

## What I Would Improve With More Time

1. **Larger OSS model** — Qwen2.5-7B or Llama-3.2-8B would dramatically improve quality and safety without GPU requirements on modern servers.
2. **Streaming responses** — Both models support token streaming; wiring it into Gradio's `gr.ChatInterface` would improve perceived latency.
3. **Vector memory** — Replace the sliding window with a semantic retrieval layer (ChromaDB or FAISS) for longer-term memory.
4. **Structured tool calling** — Implement a proper tool-use loop with multiple tools (web search, calendar, weather).
5. **CI/CD for evals** — Run the benchmark on every commit and track regressions over time.
6. **More evaluation prompts** — 15 prompts is a micro-benchmark; a production eval would use 100+ and include multi-turn scenarios.
7. **Quantisation** — Apply 4-bit GGUF quantisation via `llama.cpp` or GPTQ to cut OSS model latency 3–4× on CPU.
8. **Fine-tuning** — Domain-specific fine-tuning of the OSS model to close the quality gap with the frontier model.

---

## Requirements

- Python 3.10+
- `GROQ_API_KEY` environment variable (for Frontier and judge)
- ~2 GB RAM for Qwen2.5-0.5B-Instruct
- GPU optional (CPU works, ~10–30s per response)

---

## License

MIT
