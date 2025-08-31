# 🧠 Deep Agents: Build Powerful AI Agents for Complex Tasks

**Unleash the potential of advanced AI agents with `deepagents`, a Python package that enables you to create sophisticated agents capable of planning, executing, and adapting to solve complex, real-world problems. Check out the original repo: [https://github.com/langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)**

## Key Features

*   **Planning & Execution:** Leverages a planning tool and sub-agents to enable agents to handle complex tasks effectively.
*   **Modular Design:** Easily customize agents with your own tools, instructions, and sub-agents.
*   **Built-in Tools:** Includes essential tools like file system access (virtual), planning, and sub-agent support.
*   **Human-in-the-Loop:** Supports human approval and intervention for tool execution, enhancing control and reliability.
*   **LangGraph Integration:** Built on LangGraph, allowing seamless integration with LangGraph features (streaming, memory, etc.).
*   **MCP Compatibility:** Integrates with the Langchain MCP Adapter, enabling the use of MCP tools.
*   **Customizable Models:** Easily switch between different Language Models.

## Installation

```bash
pip install deepagents
```

## Get Started: Basic Usage

**(Requires `pip install tavily-python`)**

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

## Creating Custom Deep Agents

The `create_deep_agent` function is the core of `deepagents`. It allows you to easily configure your own deep agents.  Here's how:

### `tools` (Required)

Provide a list of functions or LangChain `@tool` objects that your agent will utilize.

### `instructions` (Required)

Set custom instructions to guide your agent's behavior.  These instructions are incorporated into the agent's prompt.

### `subagents` (Optional)

Define custom sub-agents with their own specific instructions and tools for specialized tasks:

```python
research_subagent = {
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

Customize the Language Model used by the agent.  By default, it uses `"claude-sonnet-4-20250514"`.  You can pass any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

### `builtin_tools` (Optional)

Control the built-in tools available to your agent (file system, planning). By default, these tools are all enabled.

## Deep Agent Components: Key Building Blocks

`deepagents` includes several components to facilitate deep agent behavior:

### System Prompt

A built-in system prompt provides the core instructions for planning, tool usage, and interaction with sub-agents.  This prompt is inspired by Claude Code and is designed for general use cases. This prompt can be [customized](#instructions-required).

### Planning Tool

A basic planning tool (inspired by Claude Code's TodoWrite) to assist the agent in creating and managing task plans.

### File System Tools

Includes virtual file system tools (`ls`, `edit_file`, `read_file`, `write_file`) that operate within a LangGraph State object, enabling safe and isolated file interactions.

### Sub Agents

The framework supports calling sub-agents, including a default general-purpose sub-agent. You can define [custom sub agents](#subagents-optional) with tailored instructions.

### Built-in Tools

Pre-built tools for common tasks:

*   `write_todos`: Write todos
*   `write_file`: Write to a virtual file
*   `read_file`: Read from a virtual file
*   `ls`: List files in the virtual file system
*   `edit_file`: Edit a virtual file

### Human-in-the-Loop

Integrates with human-in-the-loop approval for tool execution via the `interrupt_config` parameter, allowing for more reliable and controlled agent behavior. Currently supports `accept`, `edit` and `respond` interrupts.

## MCP Integration

Integrate with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

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

*   [ ] Allow users to customize full system prompt
*   [ ] Code cleanliness (type hinting, docstrings, formating)
*   [ ] Allow for more of a robust virtual filesystem
*   [ ] Create an example of a deep coding agent built on top of this
*   [ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)