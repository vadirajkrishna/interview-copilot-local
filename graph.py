from langgraph.graph import END, StateGraph

from agents.topic_pattern import TopicPatternAgent
from db.queries import add_exchange, add_transcript
from nodes.audio import audio_node
from nodes.framework import framework_node
from nodes.transcript import transcript_node
from state import CoachState


topic_pattern_agent = TopicPatternAgent()


async def topic_pattern_node(state: CoachState) -> CoachState:
    if state.get("framework") and state.get("steps"):
        return state

    question = state.get("question", "")
    result = await topic_pattern_agent.analyze(question)
    if not result:
        return state

    state["framework"] = result["type"]
    state["pattern"] = result.get("pattern", result["type"])
    state["steps"] = result["steps"]
    state["confidence"] = max(float(state.get("confidence", 0.0)), float(result["confidence"]))
    state["topic_model_used"] = result.get("model", "")
    state["needs_review"] = state.get("needs_review", False) and result["confidence"] < 0.6
    return state


async def persistence_node(state: CoachState) -> CoachState:
    session_id = state.get("session_id")
    if not session_id:
        return state

    await add_transcript(
        session_id=session_id,
        raw_text=state.get("raw_text", ""),
        labelled=state.get("labelled_json", {}),
    )
    if state.get("question"):
        exchange_id = await add_exchange(
            session_id=session_id,
            question=state["question"],
            answer=state.get("answer", ""),
            framework_used=state.get("framework", "General"),
        )
        state["exchange_id"] = exchange_id
    return state


def should_classify(state: CoachState) -> str:
    return "topic_pattern" if state.get("question") else "persist"


def build_graph():
    builder = StateGraph(CoachState)
    builder.add_node("audio", audio_node)
    builder.add_node("transcript", transcript_node)
    builder.add_node("topic_pattern", topic_pattern_node)
    builder.add_node("framework", framework_node)
    builder.add_node("persist", persistence_node)

    builder.set_entry_point("audio")
    builder.add_edge("audio", "transcript")
    builder.add_conditional_edges(
        "transcript",
        should_classify,
        {"topic_pattern": "topic_pattern", "persist": "persist"},
    )
    builder.add_edge("topic_pattern", "framework")
    builder.add_edge("framework", "persist")
    builder.add_edge("persist", END)
    return builder.compile()


coach_graph = build_graph()
