import asyncio
from textwrap import dedent
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import StateSnapshotEvent, EventType
class Search(BaseModel):
    query: str
    done: bool
class AgentState(BaseModel):
    searches: list[Search] = Field(default_factory=list)
agent = Agent("openai:gpt-4o-mini", deps_type=StateDeps[AgentState])

@agent.tool
async def add_search(
    ctx: RunContext[StateDeps[AgentState]], new_query: str
) -> StateSnapshotEvent:
    """Add a search to the agent's list of searches.

    Args:
    ctx: The run context containing the agent's state.
        new_query: The query to add to the agent's state.

    """
    new_search = Search(query=new_query, done=False)
    searches = ctx.deps.state.searches
    searches.append(new_search)
    agent_state = AgentState(searches=searches)
    ctx.deps.state = agent_state

    return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=agent_state)


@agent.tool
async def run_searches(ctx: RunContext[StateDeps[AgentState]]) -> StateSnapshotEvent:
    """Run the searches in the agent's state.

    Args:
        ctx: The run context containing the agent's state.
    """
    searches = ctx.deps.state.searches

    for search in searches:
        await asyncio.sleep(1)
        search.done = True

    agent_state = AgentState(searches=searches)
    ctx.deps.state = agent_state

    return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=agent_state)


@agent.instructions()
async def search_instructions(ctx: RunContext[StateDeps[AgentState]]) -> str:
    """Instructions for the search agent.

    Args:
        ctx: The run context containing search state information.

    Returns:
        Instructions string for the search agent.
    """
    return dedent(
        f"""
        You are a helpful assistant for storing searches.

        IMPORTANT:
        - Use the `add_search` tool to add a search to the agent's state
        - After using the `add_search` tool, YOU MUST ALWAYS use the `run_searches` tool to run the searches in the agent's state.
        - ONLY USE THE `add_search` TOOL ONCE FOR A GIVEN QUERY.

        The current state of the searches is:

        {ctx.deps.state.model_dump_json(indent=2)}
        """
    )


app = agent.to_ag_ui(deps=StateDeps(AgentState()))
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)