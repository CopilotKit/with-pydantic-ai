from pydantic_ai import Agent
from dotenv import load_dotenv
from ag_ui.core import CustomEvent, EventType, StateSnapshotEvent
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps

load_dotenv()

class ProverbsState(BaseModel):
    """State for the proverbs being written."""
    proverbs: list[str] = []


agent = Agent(
    'openai:gpt-4.1',
    instructions='Be fun!',
    deps_type=StateDeps[ProverbsState],
)

@agent.tool
def add_proverbs(ctx: RunContext[StateDeps[ProverbsState]], proverbs: list[str]) -> StateSnapshotEvent:
    ctx.deps.state.proverbs.extend(proverbs)
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )

@agent.tool
def get_weather(_: RunContext[StateDeps[ProverbsState]], location: str) -> str:
    """Get the weather for a given location. Ensure location is fully spelled out."""
    return f"The weather in {location} is sunny."


app = agent.to_ag_ui(deps=StateDeps(ProverbsState()))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
