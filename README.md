<!-- ---
title: InterviewCoach
colorFrom: teal
colorTo: slate
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
--- -->

# Interview Coach

Interview Coach is a local-first assistant for live technical interviews. It listens to noisy interview audio, extracts the actual Data Science, ML, AI, or System Design question, and shows a compact coaching card with important pointers while the candidate is answering.

## Why It Matters

The goal is not to handhold the candidate through the interview or generate a scripted answer. The goal is to give timely, high-signal reminders so the candidate can cover the important parts of their own answer naturally.

Interview conversations are messy: greetings, interviewer transitions, repeated words, partial transcription, candidate clarifications, and answer fragments often appear in the same transcript. Interview Coach helps by:

- Capturing the core technical question from noisy live conversation.
- Showing concise pointers that help the candidate cover the expected areas.
- Separating interviewer questions from candidate answers after the session.
- Evaluating saved Q&A exchanges with critical feedback and improvement areas.

## Multi-Model Approach

The app uses different local models for different jobs instead of asking one model to do everything:

- Speech-to-text: Whisper via `mlx-whisper` for local audio transcription.
- Topic and pattern detection: fine-tuned `vadirajkrishna/interview-coach-3b` to identify the interview question type and coarse pattern.
- Coaching hints: `Qwen/Qwen2.5-3B-Instruct` generates short, question-specific pointers for the live coaching card.
- Transcript cleanup and Q&A extraction: the general LLM extracts structured questions and candidate answers from noisy transcripts.
- Evaluation: the evaluator uses the saved candidate answer, not the coaching hints, to provide a benchmark answer, hiring band, strengths, weaknesses, and critical gaps.

This keeps the live coaching path fast while allowing more careful reasoning for post-session extraction and evaluation.

## Agentic Architecture

The app follows a LangGraph-style pipeline where each step has a focused responsibility:

```text
Audio Capture -> Transcription -> Question Extraction -> Topic/Pattern Agent
              -> Coaching Card UI

Saved Transcript -> Q&A Extraction -> SQLite Persistence -> Evaluation Agent
```

At runtime, the live path prioritizes speed: it listens to system audio, updates the transcript, extracts the latest likely technical question, classifies the question type, and renders a coaching card. After the session, the slower processing path extracts all Q&A exchanges from the full transcript and stores them in SQLite for evaluation and CSV export.

## Running Locally

Local-first interview coaching app with a Hugging Face Space demo mode.

On Hugging Face Spaces, the app uses browser microphone recording and a
Linux-compatible Transformers Whisper backend. Local-only system audio capture
via BlackHole and Ollama-based LLM calls are not available in Space mode.
