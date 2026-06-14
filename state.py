from typing import Any, TypedDict


class CoachState(TypedDict, total=False):
    session_id: int
    audio_queue: Any
    raw_text: str
    labelled_json: dict[str, Any]
    speaker: str
    question: str
    answer: str
    framework: str
    pattern: str
    steps: list[str]
    confidence: float
    needs_review: bool
    topic_model_used: str
    exchange_id: int
    evaluation: dict[str, Any]
