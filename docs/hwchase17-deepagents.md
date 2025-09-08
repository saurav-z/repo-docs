# DeepAgents: Build Powerful, Adaptive AI Agents with Python

**Unlock the potential of sophisticated AI agents capable of complex tasks.** DeepAgents empowers you to create intelligent agents that can plan, utilize tools, and operate effectively over extended processes. [Explore the DeepAgents GitHub Repo](https://github.com/hwchase17/deepagents)

## Key Features

*   **Deep Planning & Action:** Design agents that excel at tasks requiring multiple steps and strategic thinking.
*   **Tool Integration:** Seamlessly integrate various tools, from web search to file system interactions.
*   **Sub-Agent Architecture:** Leverage sub-agents for context isolation and specialized task execution.
*   **Built-in System Prompt:** Benefit from a powerful, pre-built system prompt (inspired by Claude Code) that guides agent behavior.
*   **Virtual File System:** Utilize a mock file system for secure and isolated data management within your agents.
*   **Human-in-the-Loop Support:** Integrate human oversight for critical decision-making and tool execution.
*   **Asynchronous Support:** Supports async tools and creation
*   **MCP Tools Support** Easy integration with [Langchain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
*   **Configurable Agent:** Agent creation via config file

## Installation

```bash
pip install deepagents
```

## Usage Example

Build a research agent that can perform web searches to gather information:

(Requires `pip install tavily-python`)

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

For a more complex example, refer to `examples/research/research_agent.py`.

## Customization

### `tools` (Required)

Provide a list of functions or LangChain `@tool` objects to equip your agent.

### `instructions` (Required)

Define the agent's personality and goals with custom instructions.

### `subagents` (Optional)

Create sub-agents with specific roles and instructions to enhance agent capabilities.  [See the detailed schema here.](#subagents-optional)

### `model` (Optional)

Customize the LLM used by the agent.  [See examples of using a custom model.](#example-using-a-custom-model)

### `builtin_tools` (Optional)

Control the built-in tools available to the agent. [See the list of built-in tools here.](#builtin-tools)

## Deep Agent Components

DeepAgents utilizes core components for deep task capabilities:

### System Prompt

A pre-built prompt based on the [Claude Code](https://github.com/kn1026/cc/blob/main/claudecode.md) prompt.  Customize agent behavior by modifying the [`instructions`](#instructions-required).

### Planning Tool

A basic tool to facilitate the development of a plan for the LLM to follow.

### File System Tools

Simulated file system tools (`ls`, `edit_file`, `read_file`, `write_file`) accessible through the LangGraph State object.

### Sub Agents

Built-in and custom sub-agent functionality (similar to Claude Code). Includes a `general-purpose` subagent with the same tools and instructions as the main agent.

### Built-in Tools

By default, deep agents come with five built-in tools:
- `write_todos`: Tool for writing todos
- `write_file`: Tool for writing to a file in the virtual filesystem
- `read_file`: Tool for reading from a file in the virtual filesystem
- `ls`: Tool for listing files in the virtual filesystem
- `edit_file`: Tool for editing a file in the virtual filesystem

These can be disabled via the [`builtin_tools`](#builtin-tools--optional-) parameter.

### Human-in-the-Loop

Integrate human approval for tool execution. Configure specific tools to require human approval before execution using the `interrupt_config` parameter. [See the detailed usage here.](#human-in-the-loop)

## Async Support

If you are passing async tools to your agent, you will want to `from deepagents import async_create_deep_agent`

## MCP Support

The `deepagents` library can be ran with MCP tools. This can be achieved by using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

**NOTE:** will want to use `from deepagents import async_create_deep_agent` to use the async version of `deepagents`, since MCP tools are async

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
    agent = async_create_deep_agent(tools=mcp_tools, ....)

    # Stream the agent
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is langgraph?"}]},
        stream_mode="values"
    ):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()

asyncio.run(main())
```

## Configurable Agent

Configurable agents allow you to control the agent via a config passed in.

```python
from deepagents import create_configurable_agent

agent_config = {"instructions": "foo", "subagents": []}

build_agent = create_configurable_agent(
    agent_config['instructions'],
    agent_config['subagents'],
    [],
    agent_config={"recursion_limit": 1000}
)
```
You can now use `build_agent` in your `langgraph.json` and deploy it with `langgraph dev`

For async tools, you can use `from deepagents import async_create_configurable_agent`

## Roadmap

*   \[ ] Allow users to customize full system prompt
*   \[ ] Code cleanliness (type hinting, docstrings, formating)
*   \[ ] Allow for more of a robust virtual filesystem
*   \[ ] Create an example of a deep coding agent built on top of this
*   \[ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)