import asyncio
import csv
import hashlib
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
from config import (
    APP_HOST,
    APP_PORT,
    BASE_DIR,
    GENERAL_LLM_MODEL,
    HF_SPACE_MODE,
    STREAMING_WHISPER_MODEL,
    TOPIC_PATTERN_MODEL,
)
from db.queries import (
    add_exchange,
    add_evaluation,
    add_transcript,
    append_exchange_answer,
    clear_all_tables,
    create_session,
    list_all_evaluations,
    list_evaluations,
    list_exchanges,
    update_exchange_answer,
)
from db.schema import init_db
from graph import coach_graph
from nodes.audio import LiveAudioTranscriber, transcribe_audio_array, transcribe_audio_file, warmup_transcriber
from prompts import (
    CLARIFICATION_CHECK_SYSTEM_PROMPT,
    CLARIFICATION_CHECK_USER_PROMPT,
    COACHING_GUIDANCE_SYSTEM_PROMPT,
    COACHING_GUIDANCE_USER_PROMPT,
    MULTI_EXCHANGE_EXTRACTOR_SYSTEM_PROMPT,
    MULTI_EXCHANGE_EXTRACTOR_USER_PROMPT,
    QUESTION_LIST_DETECTOR_SYSTEM_PROMPT,
    QUESTION_LIST_DETECTOR_USER_PROMPT,
    QUESTION_DETECTOR_SYSTEM_PROMPT,
    QUESTION_DETECTOR_USER_PROMPT,
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
#model_status_box textarea,
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
    max-height: 520px !important;
    overflow-y: auto !important;
    resize: vertical !important;
}
#log_box textarea,
#report_box textarea {
    min-height: 500px !important;
}
#status_box textarea {
    min-height: 34px !important;
}
#model_status_box textarea {
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

LIVE_CARD_LLM_TIMEOUT_SECONDS = 15
LIVE_QUESTION_CONTEXT_LINES = 10
LIVE_QUESTION_CONTEXT_CHARS = 2200
LIVE_QUESTION_PAUSE_SECONDS = 1.5
BROWSER_STREAM_STEP_SECONDS = 2.0
BROWSER_STREAM_CONTEXT_SECONDS = 12.0

evaluator = EvaluationAgent()
general_llm = HuggingFaceChatModel(GENERAL_LLM_MODEL)
topic_pattern_agent = TopicPatternAgent()
live_audio = LiveAudioTranscriber()
topic_model_warmup_task: asyncio.Task | None = None
model_warmup_task: asyncio.Task | None = None
model_status: dict[str, str] = {
    "General LLM": "not loaded",
    "Topic/steps model": "not loaded",
    "Speech-to-text": "not loaded",
}


def ensure_topic_model_warmup() -> None:
    global topic_model_warmup_task
    if topic_model_warmup_task and not topic_model_warmup_task.done():
        return
    topic_model_warmup_task = asyncio.create_task(
        topic_pattern_agent.analyze("How would you handle class imbalance in a fraud detection model?")
    )


def render_model_status() -> str:
    statuses = dict(model_status)
    if general_llm.is_loaded:
        statuses["General LLM"] = "loaded"
    if topic_pattern_agent.is_loaded:
        statuses["Topic/steps model"] = "loaded"
    elif not topic_pattern_agent.enabled:
        statuses["Topic/steps model"] = "disabled"
    return "\n".join(f"{name}: {status}" for name, status in statuses.items())


def all_models_ready() -> bool:
    return (
        general_llm.is_loaded
        and (topic_pattern_agent.is_loaded or not topic_pattern_agent.enabled)
        and model_status.get("Speech-to-text") == "loaded"
    )


def render_startup_status() -> str:
    if all_models_ready():
        return "All models loaded."
    return "Loading models..."


def live_detector_timeout_message(detector_name: str = "Question detector") -> str:
    if all_models_ready():
        return f"{detector_name} is taking longer than expected. Keeping the transcript live; try again in a moment."
    return f"{detector_name} is still loading. Try again after startup status shows All models loaded."


async def warmup_all_models():
    global model_warmup_task
    if model_warmup_task and not model_warmup_task.done():
        yield render_startup_status()
        return

    model_warmup_task = asyncio.current_task()
    model_status["General LLM"] = "loading"
    model_status["Topic/steps model"] = "loading" if topic_pattern_agent.enabled else "disabled"
    model_status["Speech-to-text"] = "loading"
    yield render_startup_status()

    try:
        await general_llm.warmup()
        model_status["General LLM"] = "loaded"
    except Exception as exc:
        model_status["General LLM"] = f"error: {exc}"
    yield render_startup_status()

    if topic_pattern_agent.enabled:
        try:
            await topic_pattern_agent.warmup()
            model_status["Topic/steps model"] = "loaded"
        except Exception as exc:
            model_status["Topic/steps model"] = f"error: {exc}"
    yield render_startup_status()

    try:
        await warmup_transcriber(
            model=STREAMING_WHISPER_MODEL,
            backend="transformers" if HF_SPACE_MODE else None,
        )
        model_status["Speech-to-text"] = "loaded"
    except Exception as exc:
        model_status["Speech-to-text"] = f"error: {exc}"
    yield render_startup_status() if all_models_ready() else render_model_status()


def ensure_model_warmup_background() -> None:
    global model_warmup_task
    if model_warmup_task and not model_warmup_task.done():
        return
    if all_models_ready():
        return

    async def consume_warmup() -> None:
        async for _ in warmup_all_models():
            pass

    model_warmup_task = asyncio.create_task(consume_warmup())


async def start_session(company: str, role: str) -> tuple[int, str]:
    ensure_model_warmup_background()
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
                f"{TOPIC_PATTERN_MODEL}.{suffix}"
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
        if not is_target_coaching_question(normalized["question"], item):
            continue
        cleaned.append(normalized)
    return cleaned


def dedupe_extracted_exchanges(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for exchange in exchanges:
        question = normalize_llm_text(str(exchange.get("question", "")))
        answer = normalize_llm_text(str(exchange.get("answer", "")))
        if not question or question_looks_incomplete(question):
            continue

        current = {**exchange, "question": question, "answer": answer}
        current_key = canonical_question_key(question)
        duplicate_index = -1
        for index, existing in enumerate(deduped):
            existing_key = canonical_question_key(str(existing.get("question", "")))
            if question_keys_are_similar(current_key, existing_key):
                duplicate_index = index
                break

        if duplicate_index == -1:
            deduped.append(current)
            continue

        existing = deduped[duplicate_index]
        if len(question.split()) > len(str(existing.get("question", "")).split()):
            existing["question"] = question
        if len(answer) > len(str(existing.get("answer", ""))):
            existing["answer"] = answer
    return deduped


def repair_missing_answers_from_transcript(
    exchanges: list[dict[str, Any]],
    transcript: str,
    replace_existing: bool = False,
) -> list[dict[str, Any]]:
    if not exchanges or not transcript.strip():
        return exchanges

    repaired: list[dict[str, Any]] = []
    cursor = 0
    spans: list[tuple[int, int]] = []
    for exchange in exchanges:
        span = ordered_question_match_span(transcript, str(exchange.get("question", "")), cursor)
        spans.append(span)
        if span[1] > 0:
            cursor = span[1]

    for index, exchange in enumerate(exchanges):
        item = dict(exchange)
        if str(item.get("answer", "")).strip() and not replace_existing:
            repaired.append(item)
            continue

        _, question_end = spans[index]
        if question_end < 0:
            repaired.append(item)
            continue

        next_question_start = len(transcript)
        for next_start, _ in spans[index + 1 :]:
            if next_start > question_end:
                next_question_start = next_start
                break

        answer = clean_extracted_answer_text(transcript[question_end:next_question_start])
        if answer:
            item["answer"] = answer
            item["reason"] = "Recovered answer from transcript."
        repaired.append(item)
    return repaired


def clean_extracted_answer_text(text: str) -> str:
    clean = normalize_llm_text(text.strip(" ?.:-,;"))
    if not clean:
        return ""
    clean = re.sub(r"^(good|great|okay|ok)[,.\s]+(?=(yes|i|we|my|the|in|for)\b)", "", clean, flags=re.I)
    clean = re.sub(
        r"\b(good|great|okay|ok|thank you|thanks)(?:[.!?,\s]+(?:let'?s move to the next|next question)?)?$",
        "",
        clean,
        flags=re.I,
    )
    clean = re.sub(
        r"\b(that'?s great|that is great|good|great|okay|ok)[.!?,\s]+(?:let'?s go to the next question|let'?s move to the next question|next question)[.!?,\s]*$",
        "",
        clean,
        flags=re.I,
    )
    return normalize_llm_text(clean)


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
    if looks_like_interviewer_prompt(new_question) and not question_keys_are_similar(
        canonical_question_key(previous_question),
        canonical_question_key(new_question),
    ):
        return False
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

    transcript = await transcribe_audio_input(audio_input, use_space_stt=HF_SPACE_MODE)
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

    transcript = await transcribe_audio_input(audio_input, use_space_stt=HF_SPACE_MODE)
    if transcript.startswith("[transcription unavailable:"):
        state["last_text"] = transcript
        return latest_stream_transcript(state, current_transcript), format_stream_status(state, transcript), state

    updated_transcript = (
        transcript
        if HF_SPACE_MODE
        else merge_transcript_text(latest_stream_transcript(state, current_transcript), transcript)
    )
    state["transcript"] = updated_transcript
    state["last_text"] = transcript
    state["transcriptions"] = int(state.get("transcriptions") or 0) + 1
    return updated_transcript, format_stream_status(state, "recording transcribed"), state


async def stream_live_transcript(
    session_id: int | None,
    audio_input: Any,
    current_transcript: str,
    stream_state: dict[str, Any] | None,
) -> tuple[str, str, str, dict[str, Any], dict[str, Any], str]:
    if audio_input is None:
        state = stream_state or fresh_stream_state()
        return (
            current_transcript or "",
            state.get("card_html") or render_answer_card(),
            format_stream_status(state, "waiting for microphone"),
            state,
            state.get("card_state") or {},
            await render_live_log(session_id),
        )

    state = update_stream_state(audio_input, stream_state)
    audio_buffer = state.get("audio_buffer")
    sample_rate = int(state.get("sample_rate") or 16000)
    processed_until = int(state.get("processed_until") or 0)
    available = 0 if audio_buffer is None else len(audio_buffer) - processed_until
    transcript_so_far = latest_stream_transcript(state, current_transcript)
    if HF_SPACE_MODE:
        if audio_buffer is not None:
            state["processed_until"] = len(audio_buffer)
            state["last_window_seconds"] = len(audio_buffer) / sample_rate
            state["last_rms"] = audio_rms(audio_buffer)
        return (
            transcript_so_far,
            state.get("card_html") or render_answer_card({"message": "Recording. Transcript will appear after you stop."}),
            format_stream_status(state, "recording browser audio"),
            state,
            state.get("card_state") or {},
            await render_live_log(session_id),
        )
    if available < int(sample_rate * BROWSER_STREAM_STEP_SECONDS):
        return (
            transcript_so_far,
            state.get("card_html") or render_answer_card(),
            format_stream_status(state, "buffering audio"),
            state,
            state.get("card_state") or {},
            await render_live_log(session_id),
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
        quiet_seconds = float(state.get("quiet_seconds") or 0.0) + (len(new_audio) / sample_rate)
        state["quiet_seconds"] = quiet_seconds
        if transcript_so_far:
            if quiet_seconds >= LIVE_QUESTION_PAUSE_SECONDS:
                card_html, card_state = await monitor_answer_card(
                    transcript_so_far,
                    state,
                    session_id=session_id,
                    force=False,
                    fast=True,
                    pause_check=True,
                    pause_seconds=quiet_seconds,
                )
                state["card_html"] = card_html
                state["card_state"] = card_state
            else:
                card_html = current_card_or_status(
                    state,
                    f"Pause {quiet_seconds:.1f}s. Waiting for {LIVE_QUESTION_PAUSE_SECONDS:.0f}s before checking question.",
                )
                state["card_html"] = card_html
                card_state = state.get("card_state") or {}
        else:
            card_html = state.get("card_html") or render_answer_card()
            card_state = state.get("card_state") or {}
        return (
            transcript_so_far,
            card_html,
            format_stream_status(state, "quiet audio skipped"),
            state,
            card_state,
            await render_live_log(session_id),
        )
    state["quiet_seconds"] = 0.0

    chunk_text = await transcribe_audio_array(
        sample_rate,
        audio_window,
        model=STREAMING_WHISPER_MODEL,
        backend="transformers" if HF_SPACE_MODE else None,
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
            await render_live_log(session_id),
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
            await render_live_log(session_id),
        )

    state["transcriptions"] = int(state.get("transcriptions") or 0) + 1
    state["last_text"] = chunk_text
    updated_transcript = merge_transcript_text(transcript_so_far, chunk_text)
    state["transcript"] = updated_transcript
    return (
        updated_transcript,
        state.get("card_html") or render_answer_card({"message": "Listening for a pause before creating a card."}),
        format_stream_status(state, "transcribing"),
        state,
        state.get("card_state") or {},
        await render_live_log(session_id),
    )


async def start_backend_live_transcript(session_id: int | None):
    ensure_model_warmup_background()
    await live_audio.start()
    monitor_state = fresh_card_monitor_state()
    card_html = monitor_state["card_html"]
    card_state: dict[str, Any] = {}
    card_task: asyncio.Task | None = None
    yield (
        "",
        card_html,
        "Capturing system audio via BlackHole/default input",
        card_state,
        monitor_state,
        await render_live_log(session_id),
    )
    try:
        async for transcript, pause_detected, pause_seconds in live_audio.transcript_stream():
            if card_task and card_task.done():
                try:
                    card_html, card_state = card_task.result()
                except Exception as exc:
                    monitor_state["last_extraction_message"] = f"Coaching card failed: {exc}"
                    card_html = current_card_or_status(monitor_state)
                    card_state = monitor_state.get("card_state") or {}
                card_task = None

            if pause_detected and (card_task is None or card_task.done()):
                card_task = asyncio.create_task(
                    monitor_answer_card(
                        transcript,
                        monitor_state,
                        session_id=session_id,
                        force=False,
                        fast=True,
                        pause_check=True,
                        pause_seconds=pause_seconds,
                    )
                )

            status_message = (
                "Pause detected; coaching card loading in background"
                if card_task and not card_task.done()
                else "Capturing system audio via BlackHole/default input"
            )
            yield (
                transcript,
                card_html,
                status_message,
                card_state,
                monitor_state,
                await render_live_log(session_id),
            )
    finally:
        if card_task and not card_task.done():
            card_task.cancel()
            await asyncio.gather(card_task, return_exceptions=True)


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
) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
    exchanges = dedupe_extracted_exchanges(await extract_all_interview_exchanges_with_llm(transcript))
    if exchanges:
        cards = []
        card_states = []
        last_state: dict[str, Any] = {}
        log = ""
        for exchange in exchanges:
            existing_exchange = (
                await find_existing_exchange(session_id, exchange["question"])
                if session_id
                else None
            )
            if existing_exchange:
                answer = exchange.get("answer", "").strip()
                existing_answer = str(existing_exchange.get("answer", "")).strip()
                if answer and len(answer) > len(existing_answer):
                    await update_exchange_answer(int(existing_exchange["id"]), answer)
                log = await render_log(session_id)
                display_answer = answer if answer and len(answer) > len(existing_answer) else existing_answer
                last_state = {
                    "session_id": session_id,
                    "exchange_id": existing_exchange["id"],
                    "question": existing_exchange["question"],
                    "answer": display_answer,
                    "framework": existing_exchange.get("framework_used", "General"),
                    "skipped_duplicate": True,
                }
                classification = await classify_topic_and_steps(str(existing_exchange["question"]))
                if not classification.get("model_unavailable"):
                    framework_steps = classification["steps"]
                    last_state.update(
                        {
                            "framework": classification["type"],
                            "pattern": classification.get("pattern", classification["type"]),
                            "steps": await generate_coaching_cues(
                                str(existing_exchange["question"]),
                                classification["type"],
                                classification.get("pattern", classification["type"]),
                                framework_steps,
                            ),
                            "framework_steps": framework_steps,
                            "confidence": classification["confidence"],
                        }
                    )
                    cards.append(render_card(last_state))
                    card_states.append(last_state)
                continue

            card, log, last_state = await coach_question(
                session_id,
                exchange["question"],
                exchange.get("answer", ""),
            )
            cards.append(card)
            card_states.append(last_state)
            session_id = last_state.get("session_id", session_id)
        last_state["processed_exchanges"] = exchanges
        card_html = render_cards(cards)
        monitor_state = fresh_card_monitor_state()
        monitor_state["cards"] = card_states[-4:]
        monitor_state["card_html"] = card_html
        monitor_state["card_state"] = last_state
        if card_states:
            monitor_state["last_question"] = str(card_states[-1].get("question", ""))
            monitor_state["last_question_hash"] = question_hash_key(monitor_state["last_question"])
        return transcript, card_html, log, last_state, monitor_state

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
        monitor_state = fresh_card_monitor_state()
        monitor_state["cards"] = [state]
        monitor_state["card_html"] = card
        monitor_state["card_state"] = state
        monitor_state["last_question"] = str(state.get("question", ""))
        monitor_state["last_question_hash"] = question_hash_key(monitor_state["last_question"])
        return transcript, card, log, state, monitor_state

    state = {
        "message": (
            "General LLM could not extract a complete DS/ML/AI/System Design question."
            if normalized
            else general_llm_unavailable_message()
        )
    }
    return transcript, render_answer_card(state), state["message"], state, fresh_card_monitor_state()


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
    session_id: int | None = None,
    force: bool = False,
    fast: bool = True,
    pause_check: bool = False,
    pause_seconds: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    monitor_state = monitor_state or fresh_card_monitor_state()
    if fast and not (force or pause_check):
        monitor_state["last_extraction_message"] = "Listening for the interviewer to finish the question."
        return current_card_or_status(monitor_state), monitor_state.get("card_state") or {}

    question = await detect_live_question_from_transcript(
        transcript,
        monitor_state,
        pause_seconds=pause_seconds,
    )
    if not question:
        message = monitor_state.get("last_extraction_message", "")
        card_html = current_card_or_status(monitor_state, message)
        monitor_state["card_html"] = card_html
        return card_html, monitor_state.get("card_state") or {}

    question_hash = question_hash_key(question)
    if not has_new_live_question(question, monitor_state):
        monitor_state["last_extraction_message"] = "Latest detected question is already shown."
        existing_card_state = monitor_state.get("card_state") or {}
        if session_id and existing_card_state and not existing_card_state.get("exchange_id"):
            exchange_id = await persist_live_question(session_id, transcript, existing_card_state)
            existing_card_state["session_id"] = session_id
            existing_card_state["exchange_id"] = exchange_id
            monitor_state["card_state"] = existing_card_state
        return current_card_or_status(monitor_state), monitor_state.get("card_state") or {}

    result = await classify_topic_and_steps(question)
    if result.get("model_unavailable"):
        monitor_state["last_extraction_message"] = result["message"]
        card_html = current_card_or_status(monitor_state, result["message"])
        monitor_state["card_html"] = card_html
        return card_html, {}

    cues = await generate_coaching_cues(
        question,
        result["type"],
        result.get("pattern", result["type"]),
        result["steps"],
    )
    card_state = {
        "question": question,
        "question_hash": question_hash,
        "framework": result["type"],
        "pattern": result.get("pattern", result["type"]),
        "steps": cues,
        "framework_steps": result["steps"],
        "confidence": result["confidence"],
        "needs_review": result["confidence"] < 0.6,
    }
    if session_id:
        exchange_id = await persist_live_question(session_id, transcript, card_state)
        card_state["session_id"] = session_id
        card_state["exchange_id"] = exchange_id
    cards = update_card_history(monitor_state, card_state)
    card_html = render_cards(
        [
            render_card(card, flash=card.get("question_hash") == question_hash)
            for card in cards
        ]
    )
    monitor_state["last_question"] = question
    monitor_state["last_question_hash"] = question_hash
    monitor_state["card_html"] = card_html
    monitor_state["card_state"] = card_state
    return card_html, card_state


async def persist_live_question(session_id: int, transcript: str, card_state: dict[str, Any]) -> int:
    existing_exchange_id = await find_existing_exchange_id(session_id, card_state["question"])
    if existing_exchange_id:
        await backfill_empty_exchange_answers(session_id, transcript)
        return existing_exchange_id

    await add_transcript(
        session_id=session_id,
        raw_text=transcript,
        labelled={"question": card_state["question"], "source": "live_detector"},
    )
    exchange_id = await add_exchange(
        session_id=session_id,
        question=card_state["question"],
        answer="",
        framework_used=card_state.get("framework", "General"),
    )
    await backfill_empty_exchange_answers(session_id, transcript)
    return exchange_id


async def backfill_empty_exchange_answers(session_id: int, transcript: str) -> None:
    if not transcript.strip():
        return
    exchanges = await list_exchanges(session_id)
    if not exchanges:
        return

    repair_input = [
        {
            "question": str(exchange.get("question", "")),
            "answer": str(exchange.get("answer", "")),
            "is_target": True,
            "complete": True,
            "exchange_id": exchange.get("id"),
        }
        for exchange in exchanges
    ]
    repaired = repair_missing_answers_from_transcript(repair_input, transcript)
    for original, fixed in zip(exchanges, repaired):
        if str(original.get("answer", "")).strip():
            continue
        answer = str(fixed.get("answer", "")).strip()
        if answer:
            await update_exchange_answer(int(original["id"]), answer)


async def find_existing_exchange_id(session_id: int, question: str) -> int | None:
    existing = await find_existing_exchange(session_id, question)
    return int(existing["id"]) if existing else None


async def find_existing_exchange(session_id: int, question: str) -> dict[str, Any] | None:
    question_key = canonical_question_key(question)
    exchanges = await list_exchanges(session_id)
    for exchange in exchanges:
        existing_key = canonical_question_key(str(exchange.get("question", "")))
        if question_keys_are_similar(question_key, existing_key):
            return exchange
    return None


async def render_live_log(session_id: int | None) -> str:
    if not session_id:
        return "Create a session to save live exchanges."
    return await render_log(session_id)


def current_card_or_status(state: dict[str, Any], message: str = "") -> str:
    if state.get("card_state"):
        return state.get("card_html") or render_card(state["card_state"])
    return render_answer_card({"message": message or state.get("last_extraction_message", "")})


def update_card_history(state: dict[str, Any], card_state: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    question_hash = str(card_state.get("question_hash", ""))
    cards = [
        card
        for card in state.get("cards", [])
        if isinstance(card, dict) and str(card.get("question_hash", "")) != question_hash
    ]
    cards.insert(0, card_state)
    cards = cards[:limit]
    state["cards"] = cards
    return cards


async def update_live_card_from_transcript(
    transcript: str,
    monitor_state: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    return await refresh_live_card_from_transcript(transcript, monitor_state, force=False)


async def call_coaching_from_transcript(
    session_id: int | None,
    transcript: str,
    monitor_state: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
    ensure_model_warmup_background()
    monitor_state = monitor_state or fresh_card_monitor_state()
    if not transcript.strip():
        monitor_state = fresh_card_monitor_state()
        return render_answer_card(), {}, monitor_state, await render_live_log(session_id)

    card_html, card_state = await monitor_answer_card(
        transcript,
        monitor_state,
        session_id=session_id,
        force=True,
        fast=True,
    )
    return card_html, card_state, monitor_state, await render_live_log(session_id)


async def refresh_live_card_from_transcript(
    transcript: str,
    monitor_state: dict[str, Any] | None,
    force: bool,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    monitor_state = monitor_state or fresh_card_monitor_state()
    if not force and not monitor_state.get("cards") and not monitor_state.get("last_question"):
        return gr.skip(), gr.skip(), monitor_state
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
        "last_question_hash": "",
        "card_html": render_answer_card(),
        "card_state": {},
        "cards": [],
    }


async def detect_live_question_from_transcript(
    transcript: str,
    state: dict[str, Any],
    pause_seconds: float = 0.0,
) -> str | None:
    previous_question = str(state.get("last_question") or "").strip()
    question = await detect_live_question_from_excerpt(
        recent_transcript_excerpt(transcript),
        state,
        previous_question,
        pause_seconds,
        "transcript tail",
    )
    if question or not previous_question:
        return question

    after_previous = transcript_after_question(transcript, previous_question)
    if after_previous.strip() == transcript.strip():
        return await detect_latest_question_from_list(
            recent_transcript_excerpt(transcript),
            state,
            previous_question,
        )

    question = await detect_live_question_from_excerpt(
        recent_transcript_excerpt(after_previous),
        state,
        previous_question,
        pause_seconds,
        "after previous question",
    )
    if question:
        return question

    return await detect_latest_question_from_list(
        recent_transcript_excerpt(after_previous),
        state,
        previous_question,
    )


async def detect_live_question_from_excerpt(
    excerpt: str,
    state: dict[str, Any],
    previous_question: str,
    pause_seconds: float,
    source_label: str,
) -> str | None:
    if len(excerpt.split()) < 4:
        state["last_extraction_message"] = f"Not enough transcript after {source_label} to detect a new question."
        return None

    prompt = QUESTION_DETECTOR_USER_PROMPT.format(
        transcript=excerpt,
        pause_seconds=pause_seconds,
        previous_question=previous_question or "None",
    )
    try:
        response = await asyncio.wait_for(
            general_llm.generate(
                QUESTION_DETECTOR_SYSTEM_PROMPT,
                prompt,
                max_new_tokens=256,
            ),
            timeout=LIVE_CARD_LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        state["last_extraction_message"] = live_detector_timeout_message("Question detector")
        return None

    if not response:
        details = f" Details: {general_llm.last_error}" if general_llm.last_error else ""
        state["last_extraction_message"] = (
            f"General LLM unavailable. Expected Hugging Face model {GENERAL_LLM_MODEL}.{details}"
        )
        return None

    try:
        result = json.loads(extract_json_object(response))
    except Exception as exc:
        preview = response.replace("\n", " ")[:300]
        general_llm.last_error = f"Question detector returned non-JSON output: {exc}. Output preview: {preview}"
        state["last_extraction_message"] = "Question detector returned invalid JSON."
        return None

    if not bool(result.get("question_detected")):
        if pause_seconds >= LIVE_QUESTION_PAUSE_SECONDS:
            state["last_extraction_message"] = (
                f"{pause_seconds:.1f}s pause reached, but no complete interviewer question was found in the transcript window."
            )
        else:
            state["last_extraction_message"] = "Listening for the interviewer to finish a question."
        return None

    return validate_detected_question(result, excerpt, state, source_label)


async def detect_latest_question_from_list(
    excerpt: str,
    state: dict[str, Any],
    previous_question: str,
) -> str | None:
    if len(excerpt.split()) < 4:
        return None

    prompt = QUESTION_LIST_DETECTOR_USER_PROMPT.format(
        transcript=excerpt,
        previous_question=previous_question or "None",
    )
    try:
        response = await asyncio.wait_for(
            general_llm.generate(
                QUESTION_LIST_DETECTOR_SYSTEM_PROMPT,
                prompt,
                max_new_tokens=700,
            ),
            timeout=LIVE_CARD_LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        state["last_extraction_message"] = live_detector_timeout_message("Question list detector")
        return None

    if not response:
        return None

    try:
        result = json.loads(extract_json_object(response))
    except Exception as exc:
        preview = response.replace("\n", " ")[:300]
        general_llm.last_error = f"Question list detector returned non-JSON output: {exc}. Output preview: {preview}"
        state["last_extraction_message"] = "Question list detector returned invalid JSON."
        return None

    questions = result.get("questions", [])
    if not isinstance(questions, list):
        return None

    for item in reversed(questions):
        if not isinstance(item, dict):
            continue
        question = validate_detected_question(item, excerpt, state, "question list fallback")
        if question:
            return question

    state["last_extraction_message"] = "Question list fallback found no new valid interviewer question."
    return None


def validate_detected_question(
    result: dict[str, Any],
    excerpt: str,
    state: dict[str, Any],
    source_label: str,
) -> str | None:
    speaker = str(result.get("speaker", "unknown")).strip().lower()
    confidence = str(result.get("confidence", "low")).strip().lower()
    if speaker == "candidate":
        state["last_extraction_message"] = f"Detector attributed latest text to {speaker or 'unknown'}, so no card was created."
        return None
    if confidence == "low":
        state["last_extraction_message"] = "Question detector confidence was low, so no card was created."
        return None
    if result.get("is_target") is False:
        state["last_extraction_message"] = "Detected question is outside DS/ML/AI/MLOps/System Design scope."
        return None

    question_value = result.get("question")
    question = normalize_llm_text(str(question_value)) if question_value is not None else ""
    question = trim_answer_leak_from_question(question)
    if not question:
        state["last_extraction_message"] = "Question detector did not return a question."
        return None
    if looks_like_candidate_answer_fragment(question):
        state["last_extraction_message"] = "Detector returned candidate answer text, so no coaching card was created."
        return None
    if not looks_like_interviewer_prompt(question):
        state["last_extraction_message"] = "Detector returned an answer fragment, so no coaching card was created."
        return None
    if question_looks_incomplete(question):
        state["last_extraction_message"] = "Detector returned an incomplete question fragment, so no card was created."
        return None
    if not question_is_grounded_in_transcript(question, excerpt):
        state["last_extraction_message"] = f"Detector returned a question that was not grounded in {source_label}."
        return None
    if not is_target_coaching_question(question, result):
        state["last_extraction_message"] = "Detected question is not a target technical interview question."
        return None

    question_hash = question_hash_key(question)
    if not has_new_live_question(question, state):
        state["last_extraction_message"] = f"No newer interviewer question detected in {source_label}."
        return None

    state["last_extraction_message"] = ""
    return question


def looks_like_candidate_answer_fragment(question: str) -> bool:
    clean = normalize_llm_text(question).lower().rstrip("?!. ")
    answer_starts = (
        "in supervised learning",
        "in unsupervised learning",
        "supervised learning",
        "unsupervised learning",
        "the model ",
        "a model ",
        "the key difference",
        "the main difference",
        "the main reason",
        "this means",
        "it means",
        "for example",
        "like ",
        "i would ",
        "i will ",
        "i have ",
        "i used ",
        "yes ",
        "no ",
    )
    return clean.startswith(answer_starts)


def trim_answer_leak_from_question(question: str) -> str:
    clean = normalize_llm_text(question)
    if not clean:
        return ""

    trailing_answer_tokens = (" yes", " yeah", " yep", " no", " nope", " sure", " okay", " ok")
    lowered = clean.lower().rstrip("?.!, ")
    for token in trailing_answer_tokens:
        if lowered.endswith(token):
            clean = clean[: -len(token)].rstrip(" ?.!,")
            break

    answer_starts = (
        " yes i ",
        " yes, i ",
        " yeah i ",
        " sure i ",
        " no i ",
        " i have ",
        " i've ",
        " i used ",
        " i would ",
        " i will ",
        " we used ",
        " we have ",
    )
    lowered = f" {clean.lower()} "
    cut_at = -1
    for marker in answer_starts:
        index = lowered.find(marker)
        if index > 0:
            cut_at = index
            break
    if cut_at > 0 and len(clean[:cut_at].split()) >= 5:
        clean = clean[:cut_at].rstrip(" ?.!,")

    if clean and not clean.endswith("?"):
        clean = f"{clean.rstrip('.')}?"
    return normalize_llm_text(clean)


def is_target_coaching_question(question: str, result: dict[str, Any] | None = None) -> bool:
    result = result or {}
    domain = str(result.get("domain", "")).strip().lower()
    target_domains = {
        "data_science",
        "machine_learning",
        "ai_engineering",
        "mlops",
        "statistics",
        "analytics",
        "coding",
        "algorithms",
        "system_design",
    }
    if domain in target_domains:
        return True
    if domain == "other" or result.get("is_target") is False:
        return False

    text = normalize_llm_text(question).lower()
    domain_terms = (
        "accuracy",
        "algorithm",
        "analytics",
        "api",
        "classification",
        "clustering",
        "coding",
        "data",
        "database",
        "deployment",
        "drift",
        "embedding",
        "evaluation",
        "feature",
        "fraud",
        "inference",
        "latency",
        "learning",
        "llm",
        "machine",
        "metric",
        "ml",
        "model",
        "pipeline",
        "precision",
        "production",
        "recall",
        "recommendation",
        "recommender",
        "regression",
        "scaling",
        "scalable",
        "statistics",
        "supervised",
        "system",
        "system design",
        "training",
        "unsupervised",
        "xgboost",
    )
    non_domain_terms = (
        "this app",
        "this tool",
        "how does it work here",
        "what does it mean",
        "extract relevant questions",
        "extract the required relevant questions",
        "coaching card",
        "session log",
        "transcript",
    )
    return any(term in text for term in domain_terms) and not any(term in text for term in non_domain_terms)


def recent_transcript_excerpt(
    transcript: str,
    max_lines: int = LIVE_QUESTION_CONTEXT_LINES,
    max_chars: int = LIVE_QUESTION_CONTEXT_CHARS,
) -> str:
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    excerpt = "\n".join(lines[-max_lines:]) if lines else transcript.strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[-max_chars:]
    return excerpt.strip()


def transcript_after_question(transcript: str, question: str) -> str:
    if not question:
        return transcript
    transcript_lower = transcript.lower()
    question_lower = question.lower().rstrip(" ?.")
    position = transcript_lower.rfind(question_lower)
    if position >= 0:
        return transcript[position + len(question_lower) :].strip(" ?.:-,;") or transcript

    match_end = ordered_question_match_end(transcript, question)
    if match_end > 0:
        return transcript[match_end:].strip(" ?.:-,;") or transcript
    return transcript


def ordered_question_match_span(transcript: str, question: str, start_at: int = 0) -> tuple[int, int]:
    transcript_tokens = [
        (match.group(0).lower(), match.start(), match.end())
        for match in re.finditer(r"[a-z0-9]+", transcript.lower())
        if match.end() >= start_at
    ]
    question_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if token not in {"is", "the", "a", "an", "and", "or"}
    ]
    if len(question_tokens) < 3:
        return -1, -1

    best_partial: tuple[int, int, int] | None = None
    minimum_match = max(3, int(len(question_tokens) * 0.8))
    for start_index, (token, start, end) in enumerate(transcript_tokens):
        if token != question_tokens[0]:
            continue
        question_index = 1
        matched = 1
        match_end = end
        for next_token, _, next_end in transcript_tokens[start_index + 1 :]:
            if question_index >= len(question_tokens):
                break
            if next_token == question_tokens[question_index]:
                matched += 1
                question_index += 1
                match_end = next_end
        if matched == len(question_tokens):
            return start, match_end
        if matched >= minimum_match and (
            best_partial is None or matched > best_partial[0]
        ):
            best_partial = (matched, start, match_end)

    if best_partial:
        return best_partial[1], best_partial[2]
    return -1, -1


def ordered_question_match_end(transcript: str, question: str) -> int:
    transcript_tokens = [
        (match.group(0).lower(), match.start(), match.end())
        for match in re.finditer(r"[a-z0-9]+", transcript.lower())
    ]
    question_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if token not in {"is", "the", "a", "an", "and", "or"}
    ]
    if len(question_tokens) < 3:
        return -1

    needed = len(question_tokens) if len(question_tokens) <= 6 else 6
    for start_index, (token, start, _) in enumerate(transcript_tokens):
        if token != question_tokens[0]:
            continue
        question_index = 1
        matched = 1
        end = start
        for next_token, _, next_end in transcript_tokens[start_index + 1 :]:
            if question_index >= len(question_tokens):
                break
            if next_token == question_tokens[question_index]:
                matched += 1
                question_index += 1
                end = next_end
                if matched >= needed:
                    return end
        if matched >= needed:
            return end
    return -1


def question_hash_key(question: str) -> str:
    return hashlib.md5(canonical_question_key(question).encode("utf-8")).hexdigest()


def canonical_question_key(question: str) -> str:
    text = normalize_llm_text(question).lower()
    replacements = {
        "what's": "what is",
        "whats": "what is",
        "you're": "you are",
        "you've": "you have",
        "can't": "cannot",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    words = re.findall(r"[a-z0-9]+", text)
    stop_words = {"a", "an", "the", "please", "quick", "one", "question"}
    words = [word for word in words if word not in stop_words]
    return " ".join(remove_adjacent_duplicates(words))


def remove_adjacent_duplicates(words: list[str]) -> list[str]:
    cleaned: list[str] = []
    for word in words:
        if cleaned and cleaned[-1] == word:
            continue
        cleaned.append(word)
    return cleaned


def question_looks_incomplete(question: str) -> bool:
    words = re.findall(r"[a-z0-9]+", question.lower())
    if len(words) < 5:
        return True
    return words[-1] in {"and", "or", "between", "with", "of", "to", "for", "in", "on", "the", "a", "an"}


def looks_like_interviewer_prompt(text: str) -> bool:
    clean = normalize_llm_text(text).lower()
    promptish = re.sub(r"[,:;]", "", clean)
    prompt_starts = (
        "what ",
        "what's ",
        "why ",
        "how ",
        "when ",
        "where ",
        "which ",
        "who ",
        "have you ",
        "have i ",
        "do you ",
        "did you ",
        "are you ",
        "can you ",
        "could you ",
        "would you ",
        "should you ",
        "explain ",
        "describe ",
        "tell me ",
        "walk me through ",
        "compare ",
        "design ",
        "solve ",
        "evaluate ",
        "reason about ",
        "discuss ",
        "show me ",
        "give me ",
        "build ",
        "implement ",
    )
    if promptish.startswith(prompt_starts):
        return True

    setup_starts = (
        "your model ",
        "you have ",
        "given ",
        "suppose ",
        "imagine ",
        "let's say ",
        "lets say ",
        "in production ",
        "in a production ",
        "for a ",
    )
    question_clauses = (
        " what ",
        " how ",
        " why ",
        " which ",
        " when ",
        " where ",
        " who ",
        " is this ",
        " is that ",
        " is it ",
        " are they ",
        " do you ",
        " does it ",
        " can you ",
        " could you ",
        " would you ",
        " should you ",
        " have you ",
    )
    return promptish.startswith(setup_starts) and any(clause in f" {promptish}" for clause in question_clauses)


def question_is_grounded_in_transcript(question: str, transcript: str) -> bool:
    question_tokens = content_tokens(question)
    transcript_tokens = content_tokens(transcript)
    if len(question_tokens) < 2:
        return False

    transcript_text = " ".join(transcript_tokens)
    opening = " ".join(question_tokens[:2])
    if opening and opening in transcript_text:
        return True

    if question_tokens[0] not in transcript_tokens:
        return False

    overlap = len(set(question_tokens) & set(transcript_tokens))
    return overlap / max(len(set(question_tokens)), 1) >= 0.65


def content_tokens(text: str) -> list[str]:
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on", "for", "and", "or"}
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in stop_words
    ]


def has_new_live_question(question: str, state: dict[str, Any]) -> bool:
    question_key = canonical_question_key(question)
    if not question_key:
        return False
    if question_keys_are_similar(question_key, canonical_question_key(str(state.get("last_question", "")))):
        return False
    return not any(
        question_keys_are_similar(question_key, canonical_question_key(str(card.get("question", ""))))
        for card in state.get("cards", [])
        if isinstance(card, dict)
    )


def question_keys_are_similar(left_key: str, right_key: str) -> bool:
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_words = left_key.split()
    right_words = right_key.split()
    if len(left_words) < 5 or len(right_words) < 5:
        return False
    left = set(left_words)
    right = set(right_words)
    overlap = len(left & right)
    containment = overlap / max(min(len(left), len(right)), 1)
    jaccard = overlap / max(len(left | right), 1)
    return containment >= 0.86 or jaccard >= 0.78


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
        "quiet_seconds": 0.0,
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
    model_status_box = gr.Textbox(
        label="Startup Status",
        value=render_startup_status(),
        interactive=False,
        lines=1,
        elem_id="model_status_box",
    )

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
    demo.load(
        warmup_all_models,
        outputs=[model_status_box],
        queue=True,
    )
    transcribe_coach.click(
        transcribe_and_coach,
        inputs=[session_id, mic, live_transcript],
        outputs=[live_transcript, answer_card, log, last_state],
        queue=True,
    )
    mic.stream(
        stream_live_transcript,
        inputs=[session_id, mic, live_transcript, stream_state],
        outputs=[live_transcript, answer_card, stream_status, stream_state, last_state, log],
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
            inputs=[session_id],
            outputs=[live_transcript, answer_card, stream_status, last_state, card_monitor_state, log],
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
        outputs=[live_transcript, answer_card, log, last_state, card_monitor_state],
        queue=True,
    )
    call_coaching.click(
        call_coaching_from_transcript,
        inputs=[session_id, live_transcript, card_monitor_state],
        outputs=[answer_card, last_state, card_monitor_state, log],
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
