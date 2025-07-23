import json
from pydantic_ai import Agent
from dotenv import load_dotenv
from ag_ui.core import CustomEvent, EventType, StateSnapshotEvent
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from textwrap import dedent

load_dotenv()

import logfire


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


@agent.tool_plain
async def display_proverbs(proverbs: list[str]) -> StateSnapshotEvent:
    """Display the proverbs to the user.

    Args:
        proverbs: The list of proverbs to display.

    Returns:
        StateSnapshotEvent containing the proverbs snapshot.
    """
    print("Displaying proverbs:", proverbs, flush=True)
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot={'proverbs': proverbs},
    )



# @agent.tool
# async def display_proverbs(ctx: RunContext[StateDeps[ProverbsState]]) -> StateSnapshotEvent:
#     """Display the proverbs to the user.
#     Args:
#         ctx: The run context containing proverbs state information.
#     Returns:
#         StateSnapshotEvent containing the proverbs snapshot.
#     """
#     print("Displaying proverbs:", ctx.deps.state.proverbs, flush=True)
#     return StateSnapshotEvent(
#         type=EventType.STATE_SNAPSHOT,
#         snapshot={'proverbs': ctx.deps.state.proverbs},
#     )



@agent.tool
async def set_proverbs(ctx: RunContext[StateDeps[ProverbsState]], proverbs: list[str]) -> StateSnapshotEvent:
    ctx.deps.state.proverbs = proverbs
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )



@agent.instructions
async def proverbs_instructions(ctx: RunContext[StateDeps[ProverbsState]]) -> str:
    """Instructions for the proverbs generation agent.

    Args:
        ctx: The run context containing proverbs state information.

    Returns:
        Instructions string for the proverbs generation agent.
    """

    instruct=dedent(
        f"""
        You are a helpful assistant for creating proverbs.

        IMPORTANT:
        - You will be given a list of proverbs that have already been written which may be empty.
        - Always run the `set_proverbs` tool when you start
        - Do NOT run the `set_proverbs` tool multiple times in a row
        - Use the `add_proverbs` tool to add new proverbs to the list.
        - Only add proverbs when the user explicitly asks for them.
        - Do NOT repeat the proverbs in the message, use the tool instead
        - If the user does not provide any proverbs when asking you to add to the list, make one up at random
        - If you have modified the proverbs, use the `display_proverbs` tool
        - Do NOT run the `display_proverbs` tool multiple times in a row

        Once you have completed the user's request, summarise what you did in one sentence.
        Only discuss the proverbs in the summary if you modified them.
        If you've done anything with the proverbs, do not describe them in detail or
        send them as a message to the user.

        The current state of the proverbs is:

        {json.dumps(ctx.deps.state.proverbs, indent=2)}
        """,
    )

    print("Proverbs instructions:", instruct, flush=True)
    return instruct

@agent.tool
def get_weather(_: RunContext[StateDeps[ProverbsState]], location: str) -> str:
    """Get the weather for a given location. Ensure location is fully spelled out."""
    return f"The weather in {location} is sunny."


app = agent.to_ag_ui(deps=StateDeps(ProverbsState()))

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
