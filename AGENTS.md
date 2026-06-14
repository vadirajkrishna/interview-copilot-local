# InterviewCoach — Project Brief
> Silent real-time coaching during interviews. Runs 100% local on your Mac.

---

## What It Does

When user is in an interview, the app listens to the interviewer's question via mic → transcribes it → classifies the question type → displays a compact coaching card with the framework to follow. You glance at it, then answer. No cloud. No latency. No trace.

APp will add if the question is asked by interviewer or candidate and also tags if answer is answered by the candidate or interviewer. If there is a low cofidence, flags for user updates later.

As a second part, there is another agent who goes through the transcripts and rates the interview by adding critical feedbacks highlighting areas for improvements.

---
## Architecture
LangGraph agentic pipeline. Each node is a discrete async function. State flows through the graph.
 
```
AudioNode → TranscriptNode → ClassifierAgent → FrameworkNode → UINode
                                                      ↓
                                              EvaluationAgent (post-session)
```
 
---
 
## Stack
- **Framework**: LangGraph + LangChain
- **LLM**: Qwen2.5 7B Instruct via Ollama (Part 1) / fine-tuned Qwen2.5 3B via llama.cpp (Part 2)
- **STT**: mlx-whisper (whisper-small-mlx)
- **UI**: Gradio with custom dark-mode CSS
- **Database**: SQLite (via `aiosqlite` for async)
- **Runtime**: Python 3.11+, Apple Silicon M1 Pro
---
 
## Coding Rules
- All nodes and DB calls must be `async`/`await` — no blocking I/O on the main thread
- Use `aiosqlite` for all database operations
- Use `asyncio.Queue` for audio chunk passing between nodes
- LangGraph state must be a typed `TypedDict`
- Each agent/node lives in its own file under `agents/`
---
 
## Agents
 
### ClassifierAgent
- Input: transcribed question string
- Output: `{type: str, steps: list[str]}`
- Uses Qwen2.5 7B Instruct with structured output prompt
- Must return valid framework type — fallback to "General" if uncertain
### EvaluationAgent
- Input: `{question: str, answer: str, framework: str, steps: list[str]}`
- Output: `{steps_covered: list[bool], score: int, feedback: str}`
- Runs post-session, not real-time
- One evaluation card per Q&A exchange
---
 
## Database (SQLite)
File: `interviews.db`
 
```sql
sessions    (id, date, company, role, duration)
exchanges   (id, session_id, question, answer, framework_used, timestamp)
evaluations (id, exchange_id, steps_covered_json, score, feedback)
transcripts (id, session_id, raw_text, labelled_json)
patterns    (framework, times_shown, avg_score, most_missed_step)
```
 
- All DB access via `aiosqlite`
- Init schema on app startup if tables don't exist
- Never block the event loop with synchronous sqlite3 calls
---
 
## Gradio UI
- Dark mode custom CSS — no default Gradio theme
- Three tabs: **Live** (coaching cards) · **Session Log** (transcript) · **Evaluate** (post-interview report)
- Coaching cards: colour-coded by framework type, step-by-step list
- Use `gr.Blocks` not `gr.Interface`
- UI updates via async generator / `queue=True`
---
 
## File Structure
```
interview-coach/
├── AGENTS.md
├── app.py                  # Gradio entry point
├── graph.py                # LangGraph pipeline definition
├── state.py                # TypedDict state schema
├── agents/
│   ├── classifier.py       # ClassifierAgent
│   └── evaluator.py        # EvaluationAgent
├── nodes/
│   ├── audio.py            # Whisper capture node
│   ├── transcript.py       # Speaker labelling node
│   └── framework.py        # Framework lookup node
├── db/
│   ├── schema.py           # Table definitions + init
│   └── queries.py          # Async CRUD functions
├── frameworks.yaml          # Question types + steps
├── prompts.py              # All LLM prompt templates
├── data/
│   └── train.jsonl         # Fine-tuning dataset
└── requirements.txt
```
 
---
 
## Models
- Part 1: `ollama pull qwen2.5:7b`
- Part 2: fine-tuned Qwen2.5 3B via `mlx-lm` LoRA, exported to GGUF for llama.cpp
- Swap model in one place only: `config.py` → `MODEL_PATH`
---

