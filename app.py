import asyncio
import csv
import html
import json
import os
import re
from typing import Any

if os.environ.get("SPACE_ID") or os.environ.get("INTERVIEW_COACH_RUNTIME") == "space":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gradio as gr
import numpy as np

from agents.evaluator import EvaluationAgent
from agents.hf_chat import HuggingFaceChatModel
from agents.topic_pattern import TopicPatternAgent
<<<<<<< HEAD
from config import (
    APP_HOST,
    APP_PORT,
    BASE_DIR,
    GENERAL_LLM_MODEL,
    HF_SPACE_MODE,
    STREAMING_WHISPER_MODEL,
    TOPIC_PATTERN_MODEL,
)
=======
from config import APP_HOST, APP_PORT, BASE_DIR, GENERAL_LLM_MODEL, HF_SPACE_MODE, STREAMING_WHISPER_MODEL
>>>>>>> c0f39ad (initial commit - InterviewCopilotLocal)
from db.queries import (
    add_evaluation,
    append_exchange_answer,
    clear_all_tables,
    create_session,
    list_all_evaluations,
    list_evaluations,
    list_exchanges,
)
from db.schema import init_db
from graph import coach_graph
from nodes.audio import LiveAudioTranscriber, transcribe_audio_array, transcribe_audio_file
from prompts import (
    CLARIFICATION_CHECK_SYSTEM_PROMPT,
    CLARIFICATION_CHECK_USER_PROMPT,
    COACHING_GUIDANCE_SYSTEM_PROMPT,
    COACHING_GUIDANCE_USER_PROMPT,
    MULTI_EXCHANGE_EXTRACTOR_SYSTEM_PROMPT,
    MULTI_EXCHANGE_EXTRACTOR_USER_PROMPT,
    TRANSCRIPT_NORMALIZER_REPAIR_SYSTEM_PROMPT,
    TRANSCRIPT_NORMALIZER_REPAIR_USER_PROMPT,
    TRANSCRIPT_NORMALIZER_SYSTEM_PROMPT,
    TRANSCRIPT_NORMALIZER_USER_PROMPT,
)


CSS = """
:root {
    --bg: #0b0f14;
    --panel: #111827;
    --panel-soft: #151c28;
    --panel-muted: #0f1622;
    --border: #263241;
    --border-strong: #344154;
    --text: #e7ebf0;
    --muted: #9aa5b1;
    --accent: #14b8a6;
    --accent-strong: #2dd4bf;
    --danger: #ef4444;
}
body,
.gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
}
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 12px 18px !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
#app-shell {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
#app-header {
    padding: 0;
}
#app-header h1 {
    margin: 0;
    font-size: 22px;
    line-height: 1.15;
    font-weight: 700;
}
#app-header p {
    margin: 3px 0 0;
    color: var(--muted);
    font-size: 12px;
}
.form {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--panel-muted) !important;
    padding: 8px 10px !important;
}
.action-row {
    align-items: end;
}
.action-row button,
button {
    border-radius: 6px !important;
    font-weight: 650 !important;
    min-height: 34px !important;
}
button.primary {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #041412 !important;
}
button.secondary {
    background: #1f2937 !important;
    border-color: var(--border-strong) !important;
    color: var(--text) !important;
}
button.stop {
    background: #3b1417 !important;
    border-color: #7f1d1d !important;
    color: #fecaca !important;
}
.tabs {
    border-radius: 8px !important;
}
.tab-nav {
    border-bottom: 1px solid var(--border) !important;
}
.tab-nav button {
    border-radius: 6px 6px 0 0 !important;
    color: var(--muted) !important;
    font-weight: 650 !important;
}
.tab-nav button.selected {
    color: var(--text) !important;
    border-color: var(--accent) !important;
}
.block,
.panel,
.form,
textarea,
input {
    border-color: var(--border) !important;
}
textarea,
input {
    background: #0d131d !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
label,
.label-wrap span {
    color: var(--muted) !important;
    font-weight: 650 !important;
}
#status_box textarea,
#stream_status_box textarea {
    color: var(--accent-strong) !important;
    font-size: 13px !important;
}
.section-title {
    margin: 0 0 6px;
    color: var(--muted);
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.compact-panel {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--panel-muted) !important;
    padding: 10px !important;
    min-width: 0 !important;
}
#live_grid {
    flex-wrap: nowrap !important;
    gap: 10px !important;
}
#live_grid > div {
    min-width: 0 !important;
}
#live_transcript_box textarea {
    min-height: 330px !important;
}
#log_box textarea,
#report_box textarea {
    min-height: 500px !important;
}
#status_box textarea {
    min-height: 34px !important;
}
#stream_status_box textarea {
    min-height: 72px !important;
    font-size: 12px !important;
}
.coach-card {
    border: 1px solid var(--border);
    border-left: 5px solid var(--framework-color);
    border-radius: 8px;
    background: var(--panel);
    padding: 14px;
    min-height: 425px;
    max-height: 425px;
    overflow: auto;
}
.floating-card {
    resize: both;
    min-width: 220px;
    min-height: 145px;
    max-width: 100%;
}
.coach-card summary {
    cursor: pointer;
    list-style: none;
}
.coach-card summary::-webkit-details-marker {
    display: none;
}
.card-question {
    margin: 0 0 8px;
    color: var(--text);
    font-size: 14px;
    line-height: 1.45;
    font-weight: 650;
}
.card-type {
    display: inline-block;
    margin: 4px 0 8px;
    color: var(--accent-strong);
    font-size: 13px;
    font-weight: 700;
}
.coach-card h3 {
    margin: 0 0 6px;
    font-size: 17px;
    line-height: 1.25;
}
.coach-card p {
    margin: 10px 0;
    color: var(--text);
    line-height: 1.5;
}
.coach-card ol {
    margin: 10px 0 0 22px;
    padding: 0;
    color: #d5dbe3;
}
.coach-card li {
    margin-bottom: 6px;
}
.coach-card-stack {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 10px;
    max-height: 425px;
    overflow: auto;
    padding-right: 4px;
}
.coach-card-stack .coach-card {
    min-height: 0;
    max-height: none;
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.24);
}
.meta {
    color: var(--muted);
    font-size: 13px;
}
.review {
    color: #fbbf24;
}
@keyframes card-pulse {
    0% { box-shadow: 0 0 0 0 rgba(20, 184, 166, 0.35); }
    55% { box-shadow: 0 0 0 10px rgba(20, 184, 166, 0); }
    100% { box-shadow: 0 0 0 0 rgba(20, 184, 166, 0); }
}
.flash-card {
    animation: card-pulse 1.1s ease-out infinite;
}
"""

FRAMEWORK_COLORS = {
    "General": "#64748b",
    "Behavioral": "#0ea5e9",
    "Behavioural": "#0ea5e9",
    "Technical": "#22c55e",
    "Data Science": "#22c55e",
    "AI Engineering": "#14b8a6",
    "System Design": "#f59e0b",
    "Product Sense": "#ec4899",
    "Product Design": "#ec4899",
    "Case": "#a855f7",
    "Estimation": "#a855f7",
}

GENERIC_EXTRACTION_MAX_WINDOW_CHARS = 1800
GENERIC_EXTRACTION_MIN_LLM_DELTA_CHARS = 80
LIVE_CARD_ANALYSIS_MIN_DELTA_CHARS = 30
LIVE_FAST_QUESTION_WINDOW_CHARS = 420
LIVE_FAST_MIN_DELTA_CHARS = 180
LIVE_FAST_DUPLICATE_SIMILARITY = 0.78
BROWSER_STREAM_STEP_SECONDS = 2.0
BROWSER_STREAM_CONTEXT_SECONDS = 12.0
LIVE_TARGET_FRAMEWORKS = {"Technical", "System Design"}
LIVE_QUESTION_MARKERS = (
    "first,",
    "first.",
    "first ",
    "first question",
    "first one",
    "second,",
    "second.",
    "second ",
    "second question",
    "third,",
    "third.",
    "third ",
    "third question",
    "next question",
    "another question",
    "you have",
    "let me ask",
    "can you",
    "could you",
    "tell me",
    "explain",
    "how would",
    "what would",
)
LIVE_ANSWER_STARTS = (
    " i would ",
    " i will ",
    " i can ",
    " i think ",
    " i would first ",
    " first i ",
    " my approach ",
    " so i ",
    " sure ",
    " yeah ",
    " yes ",
)
LIVE_COMPLETION_PATTERNS = (
    r"\b(is|are)\s+(this|it|that)\s+(an?\s+)?(adequate|good|bad|acceptable|enough|right|wrong)\b",
    r"\b(is|are)\s+(this|it|that).*\bmodel\b",
    r"\bwhat\s+(metric|metrics|would|should|do|is|are)\b",
    r"\bhow\s+(would|should|do|can)\b",
    r"\bwhy\s+(is|are|would|should|could|might)\b",
    r"\bshould\s+(we|you|i|the)\b",
    r"\bevaluate\b",
    r"\baccuracy\b.*\b(adequate|enough|misleading|metric|problem)\b",
)

evaluator = EvaluationAgent()
general_llm = HuggingFaceChatModel(GENERAL_LLM_MODEL)
topic_pattern_agent = TopicPatternAgent()
live_audio = LiveAudioTranscriber()
topic_model_warmup_task: asyncio.Task | None = None


def ensure_topic_model_warmup() -> None:
    global topic_model_warmup_task
    if topic_model_warmup_task and not topic_model_warmup_task.done():
        return
    topic_model_warmup_task = asyncio.create_task(
        topic_pattern_agent.analyze("How would you handle class imbalance in a fraud detection model?")
    )


async def start_session(company: str, role: str) -> tuple[int, str]:
    ensure_topic_model_warmup()
    await init_db()
    session_id = await create_session(company=company, role=role)
    return session_id, f"Session {session_id} started"


async def coach_question(session_id: int | None, question: str, answer: str) -> tuple[str, str, dict[str, Any]]:
    if not session_id:
        await init_db()
        session_id = await create_session()

    classification = await classify_topic_and_steps(question)
    if classification.get("model_unavailable"):
        state = {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "message": classification["message"],
        }
        return render_answer_card(state), classification["message"], state

    clarification_state = await handle_clarification_if_needed(session_id, question, answer)
    if clarification_state:
        return render_card(clarification_state), await render_log(session_id), clarification_state

    state = await coach_graph.ainvoke(
        {
            "session_id": session_id,
            "raw_text": question,
            "question": question,
            "answer": answer.strip(),
            "framework": classification["type"],
            "pattern": classification.get("pattern", classification["type"]),
            "steps": classification["steps"],
            "confidence": classification["confidence"],
            "topic_model_used": classification.get("topic_model_used", ""),
        }
    )
    framework_steps = state.get("steps", [])
    state["steps"] = await generate_coaching_cues(
        question,
        classification["type"],
        classification.get("pattern", classification["type"]),
        framework_steps,
    )
    state["framework_steps"] = framework_steps
    return render_card(state), await render_log(session_id), state


async def classify_topic_and_steps(question: str) -> dict[str, Any]:
    topic_result = await topic_pattern_agent.analyze(question)
    if not topic_result:
        details = topic_pattern_agent.last_error
        suffix = f" Details: {details}" if details else ""
        return {
            "type": "General",
            "steps": [],
            "confidence": 0.0,
            "model_unavailable": True,
            "message": (
                "Topic/steps model unavailable. Expected Hugging Face model "
<<<<<<< HEAD
                f"{TOPIC_PATTERN_MODEL}.{suffix}"
=======
                f"vadirajkrishna/interview-coach-3b.{suffix}"
>>>>>>> c0f39ad (initial commit - InterviewCopilotLocal)
            ),
        }

    return {
        "type": topic_result["type"],
        "steps": topic_result["steps"],
        "confidence": topic_result.get("confidence", 0.0),
        "pattern": topic_result.get("pattern", topic_result["type"]),
        "topic_model_used": topic_result.get("model", ""),
    }


async def normalize_interview_exchange_with_llm(
    transcript: str,
    question: str = "",
    answer: str = "",
    prefer_latest: bool = False,
) -> dict[str, Any] | None:
    mode_instruction = (
        "Extract the latest complete target interviewer question in the transcript window. "
        "Ignore earlier target questions that already have candidate answers. "
        "If the transcript says 'second question', 'next question', or similar, prefer the target question after that marker. "
        "In a technical/ML interview, production checks, data drift, monitoring, and training-serving data issues are target ML/MLOps questions. "
        "If first-person answer text follows the latest question, such as 'I will...' or 'I would...', keep that text in answer, not question."
        if prefer_latest
        else "Extract the clearest complete target interviewer question and its candidate answer, if present."
    )
    prompt = TRANSCRIPT_NORMALIZER_USER_PROMPT.format(
        transcript=transcript,
        mode_instruction=mode_instruction,
        question=question,
        answer=answer,
    )
    response = await general_llm.generate(
        TRANSCRIPT_NORMALIZER_SYSTEM_PROMPT,
        prompt,
        max_new_tokens=512,
    )
    if not response:
        if not general_llm.last_error:
            general_llm.last_error = "Model returned an empty response."
        return None
    try:
        payload = json.loads(extract_json_object(response))
    except Exception as exc:
        preview = response.replace("\n", " ")[:300]
        general_llm.last_error = f"Model returned non-JSON output: {exc}. Output preview: {preview}"
        return None

    normalized = normalize_normalizer_payload(payload)
    if normalized["is_target"] and normalized["complete"] and not normalized["question"]:
        repaired = await repair_normalizer_payload_with_llm(transcript, payload)
        if repaired:
            normalized = normalize_normalizer_payload(repaired)
    if (
        normalized["question"]
        and normalized["answer"]
        and normalized["answer"].lower() in normalized["question"].lower()
    ):
        repaired = await repair_normalizer_payload_with_llm(transcript, normalized)
        if repaired:
            repaired_normalized = normalize_normalizer_payload(repaired)
            if repaired_normalized["is_target"] and repaired_normalized["complete"]:
                normalized = repaired_normalized
    if normalized["is_target"] and normalized["complete"] and not normalized["question"]:
        general_llm.last_error = "Normalizer marked target complete but returned an empty question."
        normalized["is_target"] = False
        normalized["complete"] = False
        normalized["reason"] = "Target question was empty."
    return normalized


def normalize_normalizer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    clean_question = normalize_llm_text(str(payload.get("question", "")))
    clean_answer = normalize_llm_text(str(payload.get("answer", "")))
    if clean_question and not clean_question.endswith("?"):
        clean_question = f"{clean_question.rstrip('.')}?"
    return {
        "question": clean_question,
        "answer": clean_answer,
        "is_target": bool(payload.get("is_target", False)),
        "complete": bool(payload.get("complete", False)),
        "reason": str(payload.get("reason", "")).strip(),
    }


async def repair_normalizer_payload_with_llm(transcript: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    prompt = TRANSCRIPT_NORMALIZER_REPAIR_USER_PROMPT.format(
        transcript=transcript,
        payload=json.dumps(payload),
    )
    response = await general_llm.generate(
        TRANSCRIPT_NORMALIZER_REPAIR_SYSTEM_PROMPT,
        prompt,
        max_new_tokens=384,
    )
    if not response:
        return None
    try:
        return json.loads(extract_json_object(response))
    except Exception as exc:
        preview = response.replace("\n", " ")[:300]
        general_llm.last_error = f"Normalizer repair returned non-JSON output: {exc}. Output preview: {preview}"
        return None


async def extract_all_interview_exchanges_with_llm(transcript: str) -> list[dict[str, Any]]:
    prompt = MULTI_EXCHANGE_EXTRACTOR_USER_PROMPT.format(transcript=transcript)
    response = await general_llm.generate(
        MULTI_EXCHANGE_EXTRACTOR_SYSTEM_PROMPT,
        prompt,
        max_new_tokens=1400,
    )
    if not response:
        return []
    try:
        payload = json.loads(extract_json_object(response))
    except Exception as exc:
        preview = response.replace("\n", " ")[:300]
        general_llm.last_error = f"Multi-exchange extractor returned non-JSON output: {exc}. Output preview: {preview}"
        return []

    exchanges = payload.get("exchanges", [])
    if not isinstance(exchanges, list):
        return []

    cleaned = []
    for item in exchanges:
        if not isinstance(item, dict):
            continue
        normalized = normalize_normalizer_payload(item)
        if not normalized["is_target"] or not normalized["complete"] or not normalized["question"]:
            continue
        cleaned.append(normalized)
    return cleaned


def normalize_llm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" -:")


def general_llm_unavailable_message() -> str:
    details = f" Details: {general_llm.last_error}" if general_llm.last_error else ""
    return f"General LLM unavailable. Expected Hugging Face model {GENERAL_LLM_MODEL}.{details}"


async def handle_clarification_if_needed(
    session_id: int,
    question: str,
    answer: str,
) -> dict[str, Any] | None:
    exchanges = await list_exchanges(session_id)
    if not exchanges:
        return None

    previous = exchanges[-1]
    if not await is_clarification_question(previous["question"], question, answer):
        return None

    addition = f"Clarification: {question}"
    if answer.strip():
        addition = f"{addition}\nCandidate follow-up: {answer.strip()}"
    await append_exchange_answer(previous["id"], addition)

    result = await classify_topic_and_steps(previous["question"])
    cues = await generate_coaching_cues(
        previous["question"],
        previous["framework_used"],
        result.get("pattern", previous["framework_used"]),
        result["steps"],
    )
    return {
        "session_id": session_id,
        "exchange_id": previous["id"],
        "question": previous["question"],
        "answer": addition,
        "framework": previous["framework_used"],
        "pattern": result.get("pattern", previous["framework_used"]),
        "steps": cues,
        "framework_steps": result["steps"],
        "confidence": result["confidence"],
        "needs_review": False,
    }


async def is_clarification_question(previous_question: str, new_question: str, answer: str) -> bool:
    heuristic = is_clarification_question_heuristic(previous_question, new_question)
    llm_result = await is_clarification_question_with_llm(previous_question, new_question, answer)
    return llm_result if llm_result is not None else heuristic


async def is_clarification_question_with_llm(
    previous_question: str,
    new_question: str,
    answer: str,
) -> bool | None:
    prompt = CLARIFICATION_CHECK_USER_PROMPT.format(
        previous_question=previous_question,
        new_question=new_question,
        answer=answer,
    )
    response = await general_llm.generate(
        CLARIFICATION_CHECK_SYSTEM_PROMPT,
        prompt,
        max_new_tokens=256,
    )
    if not response:
        return None
    try:
        payload = json.loads(extract_json_object(response))
        return bool(payload.get("is_clarification", False))
    except Exception:
        return None


def is_clarification_question_heuristic(previous_question: str, new_question: str) -> bool:
    previous = previous_question.lower()
    current = new_question.lower().strip()
    if len(current.split()) < 3:
        return False

    clarification_starts = (
        "do you mean",
        "did you mean",
        "when you say",
        "should i assume",
        "can i assume",
        "are we assuming",
        "are we considering",
        "should we consider",
        "is it okay if",
        "can i clarify",
        "could you clarify",
        "what do you mean by",
        "for this question",
        "in this case",
    )
    if current.startswith(clarification_starts):
        return True

    clarification_terms = (
        "assume",
        "clarify",
        "constraint",
        "scope",
        "mean by",
        "considering",
        "requirement",
        "latency",
        "scale",
        "users",
        "time window",
    )
    overlap = set(re.findall(r"[a-zA-Z]+", previous)).intersection(
        set(re.findall(r"[a-zA-Z]+", current))
    )
    return any(term in current for term in clarification_terms) and bool(overlap)


async def transcribe_and_coach(
    session_id: int | None,
    audio_input: Any,
    typed_transcript: str,
) -> tuple[str, str, str, dict[str, Any]]:
    if audio_input is None:
        return "Record audio first, then press Transcribe & Coach.", render_answer_card(), "", {}

    transcript = await transcribe_audio_input(audio_input, use_space_stt=True)
    if transcript.startswith("[transcription unavailable:"):
        return transcript, render_answer_card(), "", {}

    normalized = await normalize_interview_exchange_with_llm(transcript)
    if (
        normalized
        and normalized.get("is_target")
        and normalized.get("complete")
        and normalized.get("question", "").strip()
    ):
        card, log, state = await coach_question(
            session_id,
            normalized.get("question", ""),
            normalized.get("answer", ""),
        )
        return transcript, card, log, state

    state = {
        "message": (
            "General LLM could not extract a complete DS/ML/AI/System Design question."
            if normalized
            else general_llm_unavailable_message()
        )
    }
    return transcript, render_answer_card(state), state["message"], state


async def transcribe_browser_recording(
    audio_input: Any,
    current_transcript: str,
    stream_state: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any]]:
    state = stream_state or fresh_stream_state()
    if audio_input is None:
        transcript = latest_stream_transcript(state, current_transcript)
        return transcript, format_stream_status(state, "waiting for browser recording"), state

    transcript = await transcribe_audio_input(audio_input, use_space_stt=True)
    if transcript.startswith("[transcription unavailable:"):
        state["last_text"] = transcript
        return latest_stream_transcript(state, current_transcript), format_stream_status(state, transcript), state

    updated_transcript = merge_transcript_text(latest_stream_transcript(state, current_transcript), transcript)
    state["transcript"] = updated_transcript
    state["last_text"] = transcript
    state["transcriptions"] = int(state.get("transcriptions") or 0) + 1
    return updated_transcript, format_stream_status(state, "recording transcribed"), state


async def stream_live_transcript(
    audio_input: Any,
    current_transcript: str,
    stream_state: dict[str, Any] | None,
) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
    if audio_input is None:
        state = stream_state or fresh_stream_state()
        return (
            current_transcript or "",
            state.get("card_html") or render_answer_card(),
            format_stream_status(state, "waiting for microphone"),
            state,
            state.get("card_state") or {},
        )

    state = update_stream_state(audio_input, stream_state)
    audio_buffer = state.get("audio_buffer")
    sample_rate = int(state.get("sample_rate") or 16000)
    processed_until = int(state.get("processed_until") or 0)
    available = 0 if audio_buffer is None else len(audio_buffer) - processed_until
    transcript_so_far = latest_stream_transcript(state, current_transcript)
    if available < int(sample_rate * BROWSER_STREAM_STEP_SECONDS):
        return (
            transcript_so_far,
            state.get("card_html") or render_answer_card(),
            format_stream_status(state, "buffering audio"),
            state,
            state.get("card_state") or {},
        )

    context_samples = int(sample_rate * BROWSER_STREAM_CONTEXT_SECONDS)
    end = len(audio_buffer)
    start = max(0, end - context_samples)
    audio_window = audio_buffer[start:end].copy()
    new_audio = audio_buffer[processed_until:end].copy()
    state["processed_until"] = end
    state["last_window_seconds"] = len(audio_window) / sample_rate
    rms = audio_rms(new_audio)
    state["last_rms"] = rms
    if rms < 0.003:
        return (
            transcript_so_far,
            state.get("card_html") or render_answer_card(),
            format_stream_status(state, "quiet audio skipped"),
            state,
            state.get("card_state") or {},
        )

    chunk_text = await transcribe_audio_array(
        sample_rate,
        audio_window,
        model=STREAMING_WHISPER_MODEL,
        backend="transformers",
        temperature=0.0,
        condition_on_previous_text=False,
        compression_ratio_threshold=1.8,
        logprob_threshold=-0.6,
        no_speech_threshold=0.35,
    )
    if not chunk_text or chunk_text.startswith("[transcription unavailable:"):
        return (
            transcript_so_far,
            state.get("card_html") or render_answer_card(),
            format_stream_status(state, chunk_text or "no speech detected yet"),
            state,
            state.get("card_state") or {},
        )
    if is_repetitive_hallucination(chunk_text):
        state["rejected"] = int(state.get("rejected") or 0) + 1
        state["last_text"] = f"rejected: {chunk_text[:60]}"
        return (
            transcript_so_far,
            state.get("card_html") or render_answer_card(),
            format_stream_status(state, "repeated-word output skipped"),
            state,
            state.get("card_state") or {},
        )

    state["transcriptions"] = int(state.get("transcriptions") or 0) + 1
    state["last_text"] = chunk_text
    updated_transcript = merge_transcript_text(transcript_so_far, chunk_text)
    state["transcript"] = updated_transcript
    card_html, card_state = await monitor_answer_card(updated_transcript, state, force=True)
    state["card_html"] = card_html
    state["card_state"] = card_state
    return (
        updated_transcript,
        card_html,
        format_stream_status(state, "transcribing"),
        state,
        card_state,
    )


async def start_backend_live_transcript():
    ensure_topic_model_warmup()
    await live_audio.start()
    monitor_state = fresh_card_monitor_state()
    async for transcript in live_audio.transcript_stream():
        card_html, card_state = await monitor_answer_card(
            transcript,
            monitor_state,
            force=False,
            fast=True,
        )
        yield (
            transcript,
            card_html,
            "Capturing system audio via BlackHole/default input",
            card_state,
            monitor_state,
        )


async def stop_backend_live_transcript() -> str:
    await live_audio.stop()
    return "Stopped live audio capture"


async def transcribe_audio_input(audio_input: Any, use_space_stt: bool = False) -> str:
    backend = "transformers" if use_space_stt else None
    if isinstance(audio_input, str):
        return await transcribe_audio_file(audio_input, backend=backend)
    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        sample_rate, audio = audio_input
        return await transcribe_audio_array(int(sample_rate), np.asarray(audio), backend=backend)
    return "[transcription unavailable: unsupported audio input]"


async def process_typed_transcript(
    session_id: int | None,
    transcript: str,
) -> tuple[str, str, str, dict[str, Any]]:
    exchanges = await extract_all_interview_exchanges_with_llm(transcript)
    if exchanges:
        cards = []
        last_state: dict[str, Any] = {}
        log = ""
        for exchange in exchanges:
            card, log, last_state = await coach_question(
                session_id,
                exchange["question"],
                exchange.get("answer", ""),
            )
            cards.append(card)
            session_id = last_state.get("session_id", session_id)
        last_state["processed_exchanges"] = exchanges
        return transcript, render_cards(cards), log, last_state

    normalized = await normalize_interview_exchange_with_llm(transcript)
    if (
        normalized
        and normalized.get("is_target")
        and normalized.get("complete")
        and normalized.get("question", "").strip()
    ):
        card, log, state = await coach_question(
            session_id,
            normalized.get("question", ""),
            normalized.get("answer", ""),
        )
        return transcript, card, log, state

    state = {
        "message": (
            "General LLM could not extract a complete DS/ML/AI/System Design question."
            if normalized
            else general_llm_unavailable_message()
        )
    }
    return transcript, render_answer_card(state), state["message"], state


def clear_live_state() -> tuple[str, str, dict[str, Any], str, dict[str, Any]]:
    state = fresh_stream_state()
    return "", render_answer_card(), state, format_stream_status(state, "cleared"), fresh_card_monitor_state()


async def clear_database() -> tuple[None, dict[str, Any], dict[str, Any], str, str, str, str, str, dict[str, Any]]:
    await init_db()
    await clear_all_tables()
    return (
        None,
        {},
        fresh_stream_state(),
        "All SQLite tables cleared. Start a new session to continue.",
        "",
        render_answer_card(),
        "",
        "",
        fresh_card_monitor_state(),
    )


async def monitor_answer_card(
    transcript: str,
    monitor_state: dict[str, Any] | None,
    force: bool = False,
    fast: bool = True,
) -> tuple[str, dict[str, Any]]:
    monitor_state = monitor_state or fresh_card_monitor_state()
    if fast:
        question = extract_fast_live_question_candidate(transcript, monitor_state, force=force)
    else:
        question = await extract_target_question_from_transcript(transcript, monitor_state, force=force)
    if not question:
        message = monitor_state.get("last_extraction_message", "")
        card_html = monitor_state.get("card_html") or render_answer_card({"message": message})
        return card_html, monitor_state.get("card_state") or {}

    question_key = question_dedupe_key(question)
    if is_duplicate_live_question(question_key, monitor_state):
        return monitor_state.get("card_html") or render_answer_card(), monitor_state.get("card_state") or {}
    if question_key == monitor_state.get("last_non_target_question_key"):
        return monitor_state.get("card_html") or render_answer_card(), monitor_state.get("card_state") or {}

    result = await classify_topic_and_steps(question)
    if result.get("model_unavailable"):
        monitor_state["last_extraction_message"] = result["message"]
        return monitor_state.get("card_html") or render_answer_card({"message": result["message"]}), {}
    if fast and result.get("type") not in LIVE_TARGET_FRAMEWORKS:
        monitor_state["last_non_target_question"] = question
        monitor_state["last_non_target_question_key"] = question_key
        monitor_state["last_extraction_message"] = "Listening for a DS/ML/AI/System Design question."
        return monitor_state.get("card_html") or render_answer_card({"message": monitor_state["last_extraction_message"]}), {}

    cues = await generate_coaching_cues(
        question,
        result["type"],
        result.get("pattern", result["type"]),
        result["steps"],
    )
    card_state = {
        "question": question,
        "question_key": question_key,
        "framework": result["type"],
        "pattern": result.get("pattern", result["type"]),
        "steps": cues,
        "framework_steps": result["steps"],
        "confidence": result["confidence"],
        "needs_review": result["confidence"] < 0.6,
    }
    cards = monitor_state.setdefault("cards", [])
    if not any(questions_are_similar(question_key, str(card.get("question_key", ""))) for card in cards):
        cards.append(card_state)
    card_html = render_cards([render_card(card, flash=card.get("question") == question) for card in cards])
    monitor_state["last_question"] = question
    monitor_state["last_question_key"] = question_key
    monitor_state["card_html"] = card_html
    monitor_state["card_state"] = card_state
    return card_html, card_state


async def update_live_card_from_transcript(
    transcript: str,
    monitor_state: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    return await refresh_live_card_from_transcript(transcript, monitor_state, force=False)


async def call_coaching_from_transcript(
    transcript: str,
    monitor_state: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    ensure_topic_model_warmup()
    return await refresh_live_card_from_transcript(transcript, monitor_state, force=True)


async def refresh_live_card_from_transcript(
    transcript: str,
    monitor_state: dict[str, Any] | None,
    force: bool,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    monitor_state = monitor_state or fresh_card_monitor_state()
    if not transcript.strip():
        monitor_state = fresh_card_monitor_state()
        return render_answer_card(), {}, monitor_state

    card_html, card_state = await monitor_answer_card(
        transcript,
        monitor_state,
        force=force,
        fast=True,
    )
    return card_html, card_state, monitor_state


async def generate_coaching_cues(
    question: str,
    framework: str,
    pattern: str,
    fallback_steps: list[str],
) -> list[str]:
    cues = await generate_coaching_cues_with_llm(question, framework, pattern, fallback_steps)
    if cues:
        return cues
    return fallback_steps


async def generate_coaching_cues_with_llm(
    question: str,
    framework: str,
    pattern: str,
    fallback_steps: list[str],
) -> list[str]:
    prompt = COACHING_GUIDANCE_USER_PROMPT.format(
        question=question,
        framework=framework,
        pattern=pattern,
        steps="\n".join(f"- {step}" for step in fallback_steps),
    )
    response = await general_llm.generate(
        COACHING_GUIDANCE_SYSTEM_PROMPT,
        prompt,
        max_new_tokens=384,
    )
    if not response:
        return []
    try:
        payload = json.loads(extract_json_object(response))
        cues = payload.get("cues", [])
        if isinstance(cues, list):
            return [str(cue).strip() for cue in cues if str(cue).strip()][:6]
    except Exception:
        return []
    return []


def fresh_card_monitor_state() -> dict[str, Any]:
    return {
        "last_question": "",
        "last_question_key": "",
        "card_html": render_answer_card(),
        "card_state": {},
        "cards": [],
        "last_non_target_question": "",
        "last_non_target_question_key": "",
        "generic_last_llm_until": 0,
        "generic_last_extracted_question": "",
        "fast_last_checked_until": 0,
    }


def extract_fast_live_question_candidate(
    transcript: str,
    state: dict[str, Any],
    force: bool = False,
) -> str | None:
    clean = normalize_llm_text(transcript)
    if len(clean.split()) < 5:
        state["last_extraction_message"] = "Listening for a complete DS/ML/AI/System Design question."
        return None

    last_checked_until = int(state.get("fast_last_checked_until") or 0)
    if not force and len(clean) - last_checked_until < LIVE_FAST_MIN_DELTA_CHARS:
        return None

    state["fast_last_checked_until"] = len(clean)
    candidate = clean_fast_live_question(clean[-LIVE_FAST_QUESTION_WINDOW_CHARS:])
    if len(candidate.split()) < 5:
        state["last_extraction_message"] = "Listening for a complete DS/ML/AI/System Design question."
        return None
    if not force and not live_question_looks_complete(candidate):
        state["last_extraction_message"] = "Listening for the interviewer to finish the question."
        return None

    state["last_extraction_message"] = ""
    return candidate


def clean_fast_live_question(window: str) -> str:
    candidate = f" {normalize_llm_text(window)} "
    lowered = candidate.lower()

    marker_positions = [lowered.rfind(marker) for marker in LIVE_QUESTION_MARKERS]
    marker_positions = [position for position in marker_positions if position >= 0]
    if marker_positions:
        candidate = candidate[max(marker_positions) :].strip()

    lowered = f" {candidate.lower()} "
    answer_positions = [lowered.find(marker) for marker in LIVE_ANSWER_STARTS]
    answer_positions = [position for position in answer_positions if position > 6]
    if answer_positions:
        candidate = candidate[: min(answer_positions)].strip()

    candidate = re.sub(
        r"^(?:first|second|third|next|another)(?:\s+(?:question|one))?\s*[:,.-]?\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"^(you have)\.\s+\1\b", r"\1", candidate, flags=re.IGNORECASE)

    question_mark = candidate.rfind("?")
    if question_mark >= 0:
        candidate = candidate[: question_mark + 1]

    candidate = remove_fast_transcript_repeats(candidate)
    candidate = candidate.strip(" .,-:")
    if candidate and not candidate.endswith("?"):
        candidate = f"{candidate}?"
    return candidate


def remove_fast_transcript_repeats(text: str) -> str:
    words = text.split()
    cleaned: list[str] = []
    for word in words:
        normalized = word.lower().strip(".,?!:;")
        if cleaned and normalized == cleaned[-1].lower().strip(".,?!:;"):
            continue
        cleaned.append(word)
    return " ".join(cleaned)


def live_question_looks_complete(question: str) -> bool:
    clean = normalize_llm_text(question).rstrip("?")
    lowered = clean.lower()
    if len(clean.split()) >= 22 and any(re.search(pattern, lowered) for pattern in LIVE_COMPLETION_PATTERNS):
        return True
    if len(clean.split()) >= 12 and lowered.endswith(("?", " right", " correct", " adequate", " enough")):
        return True
    return False


def question_dedupe_key(question: str) -> str:
    words = re.findall(r"[a-z0-9]+", question.lower())
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "briefly",
        "can",
        "could",
        "do",
        "does",
        "explain",
        "for",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "please",
        "question",
        "tell",
        "the",
        "to",
        "we",
        "what",
        "would",
        "you",
        "your",
    }
    useful = [word for word in words if word not in stop_words and len(word) > 1]
    return " ".join(useful[:40])


def is_duplicate_live_question(question_key: str, state: dict[str, Any]) -> bool:
    if not question_key:
        return False
    if questions_are_similar(question_key, str(state.get("last_question_key", ""))):
        return True
    return any(
        questions_are_similar(question_key, str(card.get("question_key", "")))
        for card in state.get("cards", [])
        if isinstance(card, dict)
    )


def questions_are_similar(left_key: str, right_key: str) -> bool:
    left = set(left_key.split())
    right = set(right_key.split())
    if not left or not right:
        return False
    overlap = len(left & right)
    containment = overlap / min(len(left), len(right))
    union = len(left | right)
    jaccard = overlap / union if union else 0.0
    return containment >= LIVE_FAST_DUPLICATE_SIMILARITY or jaccard >= LIVE_FAST_DUPLICATE_SIMILARITY


async def extract_target_question_from_transcript(
    transcript: str,
    state: dict[str, Any],
    force: bool = False,
) -> str | None:
    clean = re.sub(r"\s+", " ", transcript).strip()
    if len(clean.split()) < 4:
        return None

    last_llm_until = int(state.get("generic_last_llm_until") or 0)
    if (
        not force
        and len(clean) - last_llm_until < GENERIC_EXTRACTION_MIN_LLM_DELTA_CHARS
        and "?" not in clean[last_llm_until:]
    ):
        return None

    window = clean[-GENERIC_EXTRACTION_MAX_WINDOW_CHARS:]
    state["generic_last_llm_until"] = len(clean)
    result = await normalize_interview_exchange_with_llm(window, prefer_latest=True)
    if not result:
        details = f" Details: {general_llm.last_error}" if general_llm.last_error else ""
        state["last_extraction_message"] = (
            f"General LLM unavailable. Expected Hugging Face model {GENERAL_LLM_MODEL}.{details}"
        )
        return None
    if not result.get("is_target") or not result.get("complete"):
        state["last_extraction_message"] = "Listening for a complete DS/ML/AI/System Design question."
        return None

    question = str(result.get("question", "")).strip()
    if not question or question == state.get("generic_last_extracted_question"):
        return None

    state["generic_last_extracted_question"] = question
    state["last_extraction_message"] = ""
    return question


def extract_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


def merge_transcript_text(existing: str, incoming: str) -> str:
    existing = re.sub(r"\s+", " ", existing).strip()
    incoming = re.sub(r"\s+", " ", incoming).strip()
    if not existing:
        return incoming
    if not incoming:
        return existing
    if incoming.lower().startswith(existing.lower()):
        return incoming
    if existing.lower().endswith(incoming.lower()):
        return existing
    overlap = find_text_overlap(existing, incoming)
    if overlap:
        return f"{existing}{incoming[overlap:]}".strip()
    return f"{existing} {incoming}".strip()


def find_text_overlap(existing: str, incoming: str) -> int:
    existing_lower = existing.lower()
    incoming_lower = incoming.lower()
    max_overlap = min(len(existing), len(incoming), 120)
    for size in range(max_overlap, 4, -1):
        if existing_lower.endswith(incoming_lower[:size]):
            return size
    return 0


def fresh_stream_state() -> dict[str, Any]:
    return {
        "sample_rate": None,
        "audio_buffer": np.array([], dtype=np.float32),
        "transcript": "",
        "last_input_audio": np.array([], dtype=np.float32),
        "last_seen_samples": 0,
        "processed_until": 0,
        "chunks": 0,
        "transcriptions": 0,
        "rejected": 0,
        "last_rms": 0.0,
        "last_window_seconds": 0.0,
        "last_text": "",
    }


def update_stream_state(audio_input: Any, stream_state: dict[str, Any] | None) -> dict[str, Any]:
    state = stream_state or fresh_stream_state()
    if not isinstance(audio_input, tuple) or len(audio_input) != 2:
        return state

    sample_rate, audio = audio_input
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32, copy=False)

    last_seen = int(state.get("last_seen_samples") or 0)
    previous_input = state.get("last_input_audio")
    if previous_input is None:
        previous_input = np.array([], dtype=np.float32)

    if len(audio) > last_seen and audio_has_previous_prefix(audio, previous_input):
        new_audio = audio[last_seen:]
        state["last_seen_samples"] = len(audio)
    else:
        new_audio = audio
        state["last_seen_samples"] = len(audio)
    state["last_input_audio"] = audio[-int(sample_rate) * 45 :].copy()

    audio_buffer = state.get("audio_buffer")
    if audio_buffer is None:
        audio_buffer = np.array([], dtype=np.float32)

    state["sample_rate"] = int(sample_rate)
    state["chunks"] = int(state.get("chunks") or 0) + 1
    combined = np.concatenate([audio_buffer, new_audio])
    max_samples = int(sample_rate) * 45
    dropped = max(0, len(combined) - max_samples)
    state["audio_buffer"] = combined[-max_samples:]
    state["processed_until"] = max(0, int(state.get("processed_until") or 0) - dropped)
    return state


def format_stream_status(state: dict[str, Any], message: str) -> str:
    sample_rate = int(state.get("sample_rate") or 16000)
    audio_buffer = state.get("audio_buffer")
    buffered_seconds = 0.0
    if audio_buffer is not None:
        buffered_seconds = len(audio_buffer) / sample_rate
    processed_seconds = int(state.get("processed_until") or 0) / sample_rate

    chunks = int(state.get("chunks") or 0)
    transcriptions = int(state.get("transcriptions") or 0)
    rejected = int(state.get("rejected") or 0)
    rms = float(state.get("last_rms") or 0.0)
    window_seconds = float(state.get("last_window_seconds") or 0.0)
    last_text = str(state.get("last_text") or "").strip()
    if last_text:
        last_text = f" | last: {last_text[:80]}"
    return (
        f"{message} | chunks: {chunks} | buffered: {buffered_seconds:.1f}s "
        f"| processed: {processed_seconds:.1f}s | runs: {transcriptions} "
        f"| window: {window_seconds:.1f}s | rejected: {rejected} | rms: {rms:.4f}{last_text}"
    )


def latest_stream_transcript(state: dict[str, Any], current_transcript: str) -> str:
    state_transcript = str(state.get("transcript") or "").strip()
    visible_transcript = str(current_transcript or "").strip()
    if len(visible_transcript) > len(state_transcript):
        state["transcript"] = visible_transcript
        return visible_transcript
    return state_transcript


def audio_has_previous_prefix(audio: np.ndarray, previous_audio: np.ndarray) -> bool:
    if previous_audio.size == 0:
        return False
    compare_samples = min(previous_audio.size, audio.size, 8000)
    if compare_samples <= 0:
        return False
    return bool(np.allclose(audio[:compare_samples], previous_audio[:compare_samples], atol=1e-4))


def audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def is_repetitive_hallucination(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 8:
        return False

    unique_words = set(words)
    if len(unique_words) <= 2:
        return True

    most_common = max(words.count(word) for word in unique_words)
    if most_common / len(words) >= 0.65:
        return True

    repeated_run = 1
    for previous, current in zip(words, words[1:]):
        repeated_run = repeated_run + 1 if previous == current else 1
        if repeated_run >= 5:
            return True
    return False


async def evaluate_session(session_id: int | None, last_state: dict[str, Any] | None) -> str:
    if not session_id:
        return "Start a session first."

    exchanges = await list_exchanges(session_id)
    if not exchanges:
        return "No exchanges to evaluate yet."

    existing_evaluations = await list_evaluations(session_id)
    evaluated_exchange_ids = {item["exchange_id"] for item in existing_evaluations}
    pending_exchanges = [exchange for exchange in exchanges if exchange["id"] not in evaluated_exchange_ids]
    if not pending_exchanges:
        return await render_evaluations(session_id)

    latest_state_exchange_id = (last_state or {}).get("exchange_id")
    for exchange in pending_exchanges:
        if exchange["id"] == latest_state_exchange_id:
            steps = (last_state or {}).get("framework_steps") or (last_state or {}).get("steps") or []
        else:
            classification = await classify_topic_and_steps(exchange["question"])
            steps = classification.get("steps", [])
        if not steps:
            steps = ["Clarify", "Answer", "Example", "Impact"]

        result = await evaluator.evaluate(
            question=exchange["question"],
            answer=exchange["answer"],
            framework=exchange["framework_used"],
            steps=steps,
        )
        await add_evaluation(
            exchange_id=exchange["id"],
            steps_covered=result["steps_covered"],
            score=result["score"],
            feedback=result["feedback"],
        )
    return await render_evaluations(session_id)


async def render_log(session_id: int) -> str:
    exchanges = await list_exchanges(session_id)
    if not exchanges:
        return "No exchanges yet."
    lines = []
    for item in exchanges:
        lines.append(
            f"Q{item['id']} [{item['framework_used']}]: {item['question']}\n"
            f"A: {item['answer'] or '(not captured yet)'}"
        )
    return "\n\n".join(lines)


async def render_evaluations(session_id: int) -> str:
    evaluations = await list_evaluations(session_id)
    if not evaluations:
        return "No evaluations yet."
    blocks = []
    for item in evaluations:
        blocks.append(
            f"Exchange {item['exchange_id']}\n"
            f"Framework: {item['framework_used']}\n"
            f"{item['feedback']}"
        )
    return "\n\n".join(blocks)


async def render_sqlite_evaluation_summary(session_id: int | None) -> tuple[str, list[list[str]]]:
    evaluations = await list_evaluations(session_id) if session_id else await list_all_evaluations()
    if not evaluations:
        return "No evaluated exchanges found in SQLite. Run Evaluate Session first.", []

    blocks = []
    rows = []
    for item in evaluations:
        feedback_parts = split_evaluation_feedback(item["feedback"])
        session_name = format_session_name(item)
        session_date = format_session_date(item.get("date", ""))
        session_label = ""
        if item.get("session_id"):
            session_label = f"{session_name} - {session_date}\n"
        blocks.append(
            f"{session_label}"
            f"Exchange {item['exchange_id']} [{item['framework_used']}]\n\n"
            f"Question:\n{item['question']}\n\n"
            f"My answer:\n{item['answer'] or '(not captured yet)'}\n\n"
            f"Benchmark answer:\n{feedback_parts['benchmark']}\n\n"
            f"Evaluation band:\n{feedback_parts['band']}\n\n"
            f"Feedback:\n{feedback_parts['feedback']}"
        )
        rows.append(
            [
                session_name,
                session_date,
                item["question"],
                item["answer"] or "(not captured yet)",
                feedback_parts["benchmark"],
                feedback_parts["band"],
                feedback_parts["feedback"],
            ]
        )
    return "\n\n---\n\n".join(blocks), rows


async def export_sqlite_evaluation_summary_csv(session_id: int | None) -> str | None:
    _, rows = await render_sqlite_evaluation_summary(session_id)
    if not rows:
        return None

    export_dir = BASE_DIR / ".runtime" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "evaluation_summary.csv"
    with export_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(EVALUATION_TABLE_HEADERS)
        writer.writerows(rows)
    return str(export_path)


EVALUATION_TABLE_HEADERS = [
    "Session",
    "Date",
    "Question",
    "Candidate Answer",
    "Model Benchmark Answer",
    "Evaluation Band",
    "Feedback",
]


def format_session_name(item: dict[str, Any]) -> str:
    session_id = item.get("session_id", "")
    company = item.get("company") or "Unknown company"
    role = item.get("role") or "Unknown role"
    return f"Session {session_id} - {company} / {role}"


def format_session_date(value: str) -> str:
    if not value:
        return ""
    return value.replace("T", " ").split(".")[0]


def split_evaluation_feedback(feedback: str) -> dict[str, str]:
    benchmark = extract_feedback_section(feedback, "Agent benchmark answer:", "Evaluation band:")
    if not benchmark:
        benchmark = extract_feedback_section(feedback, "Agent benchmark answer:", "Score:")

    band = extract_feedback_section(feedback, "Evaluation band:", "Strong points:")
    score_and_feedback = extract_feedback_section(feedback, "Evaluation band:", "")
    if band:
        score_and_feedback = extract_feedback_section(feedback, "Strong points:", "")
        score_and_feedback = f"Strong points:\n{score_and_feedback}".strip()
    else:
        score_and_feedback = extract_feedback_section(feedback, "Score:", "")
        if score_and_feedback:
            score_text = re.match(r"^\s*(\d+\s*/\s*5)", score_and_feedback)
            band = score_text.group(1) if score_text else "Unknown"
            score_and_feedback = re.sub(r"^\s*\d+\s*/\s*5\s*", "", score_and_feedback).strip()
    return {
        "benchmark": benchmark or "(benchmark not available)",
        "band": band or "Unknown",
        "feedback": score_and_feedback or feedback.strip() or "(feedback not available)",
    }


def extract_feedback_section(text: str, start_label: str, end_label: str) -> str:
    start = text.find(start_label)
    if start == -1:
        return ""
    start += len(start_label)
    end = text.find(end_label, start) if end_label else -1
    if end == -1:
        return text[start:].strip()
    return text[start:end].strip()


def render_card(state: dict[str, Any], flash: bool = False) -> str:
    framework = html.escape(state.get("framework", "General"))
    pattern = html.escape(state.get("pattern", ""))
    display_type = pattern or framework
    question = html.escape(state.get("question", "Question unavailable"))
    color = FRAMEWORK_COLORS.get(display_type, FRAMEWORK_COLORS.get(framework, FRAMEWORK_COLORS["General"]))
    steps = "".join(f"<li>{html.escape(step)}</li>" for step in state.get("steps", []))
    flash_class = " flash-card" if flash else ""
    return f"""
    <details class="coach-card floating-card{flash_class}" style="--framework-color: {color}" open>
        <summary>
            <div class="card-question">{question}</div>
            <span class="card-type">Type: {display_type}</span>
        </summary>
        <div class="meta">Steps</div>
        <ol>{steps}</ol>
    </details>
    """


def render_cards(cards: list[str]) -> str:
    if not cards:
        return render_answer_card()
    return "<div class='coach-card-stack'>" + "\n".join(cards) + "</div>"


def render_answer_card(state: dict[str, Any] | None = None) -> str:
    state = state or {}
    status = state.get("message") or "Waiting for answer card content."
    if state.get("answer"):
        status = "Answer captured. Card content to be defined."
    return f"""
    <div class="coach-card" style="--framework-color: #22c55e">
        <h3>Answer Card</h3>
        <div class="meta">{html.escape(status)}</div>
    </div>
    """


with gr.Blocks(elem_id="app-shell") as demo:
    session_id = gr.State(None)
    last_state = gr.State({})
    stream_state = gr.State(fresh_stream_state())
    card_monitor_state = gr.State(fresh_card_monitor_state())

    runtime_label = "Hugging Face Space browser-mic demo" if HF_SPACE_MODE else "Local and Space-style audio coaching"
    gr.Markdown(
        f"# InterviewCoach\n{runtime_label} with live transcript, framework cards, and post-session evaluation.",
        elem_id="app-header",
    )
    with gr.Row(elem_classes=["form", "action-row"]):
        company = gr.Textbox(label="Company", placeholder="Anthropic", scale=2)
        role = gr.Textbox(label="Role", placeholder="ML Engineer", scale=2)
        start = gr.Button("Create Session", variant="primary", scale=1)
    status = gr.Textbox(label="Session Status", interactive=False, elem_id="status_box")

    with gr.Tabs():
        with gr.Tab("Live"):
            with gr.Row(equal_height=True, elem_id="live_grid"):
                with gr.Column(scale=2, min_width=180, elem_classes=["compact-panel"]):
                    gr.Markdown("Space-style browser mic", elem_classes=["section-title"])
                    mic = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Record from browser",
                        streaming=True,
                        elem_id="mic_box",
                    )
                    with gr.Row(elem_classes=["action-row"]):
                        transcribe_coach = gr.Button("Process Recording", variant="primary")

                    if not HF_SPACE_MODE:
                        gr.Markdown("System audio", elem_classes=["section-title"])
                        with gr.Row(elem_classes=["action-row"]):
                            start_live = gr.Button("Start System Audio", variant="secondary")
                            stop_live = gr.Button("Stop System Audio", variant="stop")
                    with gr.Row(elem_classes=["action-row"]):
                        clear = gr.Button("Clear Screen", variant="secondary")
                    stream_status = gr.Textbox(
                        label="Audio Status",
                        interactive=False,
                        lines=2,
                        elem_id="stream_status_box",
                    )

                with gr.Column(scale=3, min_width=280, elem_classes=["compact-panel"]):
                    gr.Markdown("Transcript workspace", elem_classes=["section-title"])
                    live_transcript = gr.Textbox(
                        label="Live Transcript",
                        lines=11,
                        placeholder="Recorded or typed transcript appears here.",
                        elem_id="live_transcript_box",
                    )
                    with gr.Row(elem_classes=["action-row"]):
                        call_coaching = gr.Button("Call Coaching", variant="primary")
                        coach = gr.Button("Process Text", variant="secondary")

                with gr.Column(scale=2, min_width=220, elem_classes=["compact-panel"]):
                    gr.Markdown("Coaching card", elem_classes=["section-title"])
                    answer_card = gr.HTML(render_answer_card())

        with gr.Tab("Session Log"):
            log = gr.Textbox(label="Saved Exchanges", lines=16, interactive=False, elem_id="log_box")
            clear_db = gr.Button("Clear SQLite Tables", variant="stop")

        with gr.Tab("Evaluate"):
            with gr.Row(elem_classes=["action-row"]):
                evaluate = gr.Button("Evaluate Session", variant="primary")
                load_eval_summary = gr.Button("Load SQLite Summary", variant="secondary")
                export_eval_csv = gr.Button("Export CSV", variant="secondary")
            evaluation_table = gr.Dataframe(
                headers=EVALUATION_TABLE_HEADERS,
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                label="SQLite Evaluation History",
                interactive=False,
                wrap=True,
            )
            export_file = gr.File(label="CSV Download", interactive=False)
            report = gr.Textbox(label="Evaluation Report", lines=16, interactive=False, elem_id="report_box")

    start.click(start_session, inputs=[company, role], outputs=[session_id, status])
    transcribe_coach.click(
        transcribe_and_coach,
        inputs=[session_id, mic, live_transcript],
        outputs=[live_transcript, answer_card, log, last_state],
        queue=True,
    )
    mic.stream(
        stream_live_transcript,
        inputs=[mic, live_transcript, stream_state],
        outputs=[live_transcript, answer_card, stream_status, stream_state, last_state],
        queue=True,
    )
    mic.stop_recording(
        transcribe_browser_recording,
        inputs=[mic, live_transcript, stream_state],
        outputs=[live_transcript, stream_status, stream_state],
        queue=True,
    )
    if not HF_SPACE_MODE:
        start_live.click(
            start_backend_live_transcript,
            outputs=[live_transcript, answer_card, stream_status, last_state, card_monitor_state],
            queue=True,
        )
        stop_live.click(
            stop_backend_live_transcript,
            outputs=[stream_status],
            queue=False,
        )
    coach.click(
        process_typed_transcript,
        inputs=[session_id, live_transcript],
        outputs=[live_transcript, answer_card, log, last_state],
        queue=True,
    )
    call_coaching.click(
        call_coaching_from_transcript,
        inputs=[live_transcript, card_monitor_state],
        outputs=[answer_card, last_state, card_monitor_state],
        queue=True,
    )
    live_transcript.change(
        update_live_card_from_transcript,
        inputs=[live_transcript, card_monitor_state],
        outputs=[answer_card, last_state, card_monitor_state],
        queue=True,
    )
    clear.click(
        clear_live_state,
        outputs=[live_transcript, answer_card, stream_state, stream_status, card_monitor_state],
        queue=False,
    )
    clear_db.click(
        clear_database,
        outputs=[
            session_id,
            last_state,
            stream_state,
            status,
            live_transcript,
            answer_card,
            log,
            report,
            card_monitor_state,
        ],
        queue=True,
    )
    evaluate.click(evaluate_session, inputs=[session_id, last_state], outputs=[report], queue=True)
    load_eval_summary.click(
        render_sqlite_evaluation_summary,
        inputs=[session_id],
        outputs=[report, evaluation_table],
        queue=True,
    )
    export_eval_csv.click(
        export_sqlite_evaluation_summary_csv,
        inputs=[session_id],
        outputs=[export_file],
        queue=True,
    )


if __name__ == "__main__":
    port = int(os.environ.get("INTERVIEW_COACH_PORT", APP_PORT))
    demo.queue().launch(
        css=CSS,
        theme=gr.themes.Base(),
        server_name=APP_HOST,
        server_port=port,
    )
