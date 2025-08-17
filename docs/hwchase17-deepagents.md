# DeepAgents: Build Advanced LLM Agents for Complex Tasks

**Unlock the power of sophisticated LLM agents with DeepAgents, enabling planning, sub-agents, and file system interaction for in-depth research and problem-solving.**  [View the original repo](https://github.com/hwchase17/deepagents)

<img src="deep_agents.png" alt="deep agent" width="600"/>

DeepAgents is a Python package designed to streamline the creation of advanced language model (LLM) agents, allowing you to build agents that excel at complex tasks. Inspired by projects like "Claude Code," DeepAgents simplifies the implementation of planning, sub-agents, and file system access, enabling your agents to tackle intricate challenges effectively.

## Key Features:

*   **Planning Tool:** Built-in tool inspired by Claude Code for creating and managing task plans.
*   **Sub-Agent Support:** Enables the use of specialized sub-agents for focused tasks and context isolation.
*   **Virtual File System:** Includes file system tools ( `ls`, `edit_file`, `read_file`, `write_file`) to simulate file interaction, allowing agents to manage and process information.
*   **Customizable Instructions:** Provides flexibility in crafting agent behavior through prompt customization.
*   **LangGraph Integration:** Built upon LangGraph, allowing you to integrate with streaming, human-in-the-loop, memory, and studio capabilities.
*   **MCP Tool Compatibility:** Supports integration with MCP tools for enhanced functionality.

## Installation

```bash
pip install deepagents
```

## Usage

(To run the example below, will need to `pip install tavily-python`)

```python
import os
from typing import Literal

from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# Create the agent
agent = create_deep_agent(
    [internet_search],
    research_instructions,
)

# Invoke the agent
result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
```

See [examples/research/research_agent.py](examples/research/research_agent.py) for a more complex example.

## Creating a Custom Deep Agent

Customize your agent's behavior with the `create_deep_agent` function. This function takes three primary parameters:

### `tools` (Required)

A list of functions or LangChain `@tool` objects that the agent and its sub-agents will utilize.

### `instructions` (Required)

A prompt that helps guide agent behavior and provide it with the right context. A built-in system prompt is also included.

### `subagents` (Optional)

Define custom sub-agents using dictionaries that have the following schema:

```python
class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
```

-   **name**: The name the main agent uses to call the subagent
-   **description**: A description of the subagent, shown to the main agent.
-   **prompt**: The prompt for the subagent.
-   **tools**: The tools the subagent has access to. Defaults to all tools passed to the main agent.

Example:

```python
research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "prompt": sub_research_prompt,
}
subagents = [research_subagent]
agent = create_deep_agent(
    tools,
    prompt,
    subagents=subagents
)
```

### `model` (Optional)

Customize the LLM used by the agent.  By default, `"claude-sonnet-4-20250514"` is used. Pass in any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

#### Example: Using a Custom Model (Ollama)

(Requires `pip install langchain` and then `pip install langchain-ollama` for Ollama models)

```python
from deepagents import create_deep_agent

# ... existing agent definitions ...

model = init_chat_model(
    model="ollama:gpt-oss:20b",  
)
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
    ...
)
```

## Deep Agent Details

### System Prompt

DeepAgents includes a comprehensive built-in system prompt. It is inspired by Claude Code, but made more general purpose and is the foundation for effective agent performance. The prompt manages tool usage, planning, file system interactions, and sub-agent orchestration. Part of this prompt is customizable via the `instructions` parameter.

### Planning Tool

Based on ClaudeCode's TodoWrite tool, this tool is a crucial part of the agent's architecture, assisting in task planning.

### File System Tools

Provides file system tools ( `ls`, `edit_file`, `read_file`, `write_file`) using LangGraph's State object to simulate a virtual file system. Files are accessed using the `files` key within the LangGraph State object.

### Sub Agents

Supports the use of sub-agents (inspired by Claude Code), facilitating context quarantine and customized instructions. It also includes a `general-purpose` subagent by default.

## MCP Integration

The `deepagents` library works with MCP tools and can be integrated using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

(To run the example below, will need to `pip install langchain-mcp-adapters`)

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

async def main():
    # Collect MCP tools
    mcp_client = MultiServerMCPClient(...)
    mcp_tools = await mcp_client.get_tools()

    # Create agent
    agent = create_deep_agent(tools=mcp_tools, ....)

    # Stream the agent
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is langgraph?"}]},
        stream_mode="values"
    ):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()

asyncio.run(main())
```

## Roadmap

*   \[ ] Allow users to customize full system prompt
*   \[ ] Code cleanliness (type hinting, docstrings, formating)
*   \[ ] Allow for more of a robust virtual filesystem
*   \[ ] Create an example of a deep coding agent built on top of this
*   \[ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)
*   \[ ] Add human-in-the-loop support for tools