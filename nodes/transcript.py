from state import CoachState


QUESTION_HINTS = ("?", "tell me", "how would", "can you", "describe", "design")


async def transcript_node(state: CoachState) -> CoachState:
    raw_text = state.get("raw_text", "").strip()
    if state.get("question"):
        speaker = "interviewer"
    elif state.get("answer"):
        speaker = "candidate"
    else:
        speaker = infer_speaker(raw_text)
    labelled = {
        "speaker": speaker,
        "text": raw_text,
        "confidence": 0.65 if raw_text else 0.0,
    }

    state["speaker"] = speaker
    state["labelled_json"] = labelled
    if speaker == "interviewer" and not state.get("question"):
        state["question"] = raw_text
    elif speaker == "candidate" and not state.get("answer"):
        state["answer"] = raw_text
    state["needs_review"] = labelled["confidence"] < 0.6
    return state


def infer_speaker(text: str) -> str:
    lowered = text.lower()
    if text.endswith("?") or any(hint in lowered for hint in QUESTION_HINTS):
        return "interviewer"
    return "candidate"
