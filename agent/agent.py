import json
from pydantic_ai import Agent
from dotenv import load_dotenv
from ag_ui.core import CustomEvent, EventType, StateSnapshotEvent
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from textwrap import dedent

load_dotenv()

class ProverbsState(BaseModel):
    """List of the proverbs being written."""
    proverbs: list[str] = Field(
        default_factory=list,
        description='The list of already written proverbs',
    )

agent = Agent(
    'openai:gpt-4.1',
    deps_type=StateDeps[ProverbsState],
)

@agent.tool
async def add_proverbs(ctx: RunContext[StateDeps[ProverbsState]], proverbs: list[str]) -> StateSnapshotEvent:
    ctx.deps.state.proverbs.extend(proverbs)
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )

@agent.tool
async def set_proverbs(ctx: RunContext[StateDeps[ProverbsState]], proverbs: list[str]) -> StateSnapshotEvent:
    ctx.deps.state.proverbs = proverbs
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )

@agent.tool
async def display_proverbs(ctx: RunContext[StateDeps[ProverbsState]]) -> StateSnapshotEvent:
    """Display the proverbs to the user.

    Args:
        ctx: The run context containing proverbs state information.

    Returns:
        StateSnapshotEvent containing the proverbs snapshot.
    """
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot={'proverbs': ctx.deps.state.proverbs},
    )

@agent.instructions
async def proverbs_instructions(ctx: RunContext[StateDeps[ProverbsState]]) -> str:
    """Instructions for the proverbs generation agent.

    Args:
        ctx: The run context containing proverbs state information.

    Returns:
        Instructions string for the proverbs generation agent.
    """

    return dedent(
        f"""
        You are a helpful assistant for creating proverbs.

        IMPORTANT:
        - Create a list of proverbs based on the user's input.
        - Use the `add_proverbs` tool to add new proverbs to the list.
        - After adding proverbs, use the `display_proverbs` tool to show the updated list to the user.
        - Use the `display_proverbs` tool to present the list of proverbs to the user
        - Always run the set_proverbs tool when you start
        - Do NOT run the set_proverbs tool multiple times in a row
        - Do NOT repeat the proverbs in the message, use the tool instead
        - Do NOT run the `display_proverbs` tool multiple times in a row
        - If the user does not provide any proverbs, make one up at random
        - Even if the user requests for you to do something you cannot do, finish by running display_proverbs

        Once you have created the updated list of proverbs and displayed it to the user,
        summarise the changes in one sentence, don't describe the proverbs in
        detail or send it as a message to the user.

        The current state of the proverbs is:

        {json.dumps(ctx.deps.state.proverbs, indent=2)}
        """,
    )

@agent.tool
def get_weather(_: RunContext[StateDeps[ProverbsState]], location: str) -> str:
    """Get the weather for a given location. Ensure location is fully spelled out."""
    return f"The weather in {location} is sunny."


app = agent.to_ag_ui(deps=StateDeps(ProverbsState()))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
