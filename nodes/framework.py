import yaml

from state import CoachState


async def framework_node(state: CoachState) -> CoachState:
    framework = state.get("framework", "General")
    with open("frameworks.yaml", "r", encoding="utf-8") as file:
        frameworks = yaml.safe_load(file)

    if framework not in frameworks:
        framework = "General"

    state["framework"] = framework
    if not state.get("steps"):
        state["steps"] = frameworks[framework]["steps"]
    return state
