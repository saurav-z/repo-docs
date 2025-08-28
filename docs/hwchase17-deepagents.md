# Deep Agents: Build Powerful, Autonomous AI Agents with Ease

**Unlock the potential of sophisticated AI agents capable of complex tasks with `deepagents`, a Python package that simplifies the creation of intelligent, multi-step agents.** [View the original repo](https://github.com/hwchase17/deepagents)

## Key Features

*   **Modular Design:** Easily integrate planning, sub-agents, file system interaction, and detailed prompting to create deep agents.
*   **Built-in Planning Tool:** The agent can plan and execute complex tasks using a built-in planning tool, inspired by the Claude Code's TodoWrite tool.
*   **Virtual File System:** Utilize built-in file system tools (`ls`, `edit_file`, `read_file`, `write_file`) for seamless data management within your agent's workflow, based on LangGraph's State object.
*   **Sub-Agent Support:** Employ sub-agents for context management and specialized task handling.
*   **Customizable Tools:** Leverage pre-built tools and integrate your own functions or LangChain `@tool` objects.
*   **Human-in-the-Loop Interrupts:** Implement human approval for tool execution using the `interrupt_config` parameter.
*   **Built-in System Prompt:** Benefit from a robust system prompt inspired by Claude Code, optimized for deep agent performance.
*   **LangGraph Integration:** Built on LangGraph, providing compatibility with streaming, human-in-the-loop, memory, and studio features.
*   **MCP Integration:** Run with MCP tools via the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

## Installation

```bash
pip install deepagents
```

## Getting Started

Build your own agent by importing the `create_deep_agent` function. Here's a quick example using an internet search tool:

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

## Customizing Your Deep Agent

`create_deep_agent` allows you to customize agent behavior through a set of parameters:

### `tools` (Required)

A list of functions or LangChain `@tool` objects the agent will have access to.

### `instructions` (Required)

A prompt that forms part of the agent's overall instructions.

### `subagents` (Optional)

Define custom sub-agents with their own instructions and tools.  See the original README for a code example and a full description of `SubAgent` options.

### `model` (Optional)

Specify a custom LangChain model (defaults to `"claude-sonnet-4-20250514"`).

#### Example: Using a Custom Model (Ollama with gpt-oss)

```python
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama  # Requires pip install langchain-ollama

model = ChatOllama(model="gpt-oss:20b")

agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
)
```

#### Example: Per-subagent model override (optional)

```python
from deepagents import create_deep_agent

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Critique the final report",
    "prompt": "You are a tough editor.",
    "model_settings": {
        "model": "anthropic:claude-3-5-haiku-20241022",
        "temperature": 0,
        "max_tokens": 8192
    }
}

agent = create_deep_agent(
    tools=[internet_search],
    instructions="You are an expert researcher...",
    model="claude-sonnet-4-20250514",  # default for main agent and other sub-agents
    subagents=[critique_sub_agent],
)
```

### `builtin_tools` (Optional)

Control which built-in tools are available to the agent.

```python
# Only give agent access to todo tool, none of the filesystem tools
builtin_tools = ["write_todos"]
agent = create_deep_agent(..., builtin_tools=builtin_tools, ...)
```

## Deep Agent Components

### System Prompt

A detailed, customizable system prompt designed for deep agent performance, inspired by Claude Code.

### Planning Tool

A simple planning tool (based on ClaudeCode's TodoWrite tool) to help the agent strategize.

### File System Tools

Simulate a virtual file system using `ls`, `edit_file`, `read_file`, and `write_file` for managing agent context.

### Sub Agents

Enable the creation and utilization of sub-agents for context management and specialized instruction.

### Built-In Tools

Pre-built tools to simplify common tasks: `write_todos`, `write_file`, `read_file`, `ls`, and `edit_file`.

### Tool Interrupts

Configure human-in-the-loop approval for tool execution.

```python
from deepagents import create_deep_agent
from langgraph.prebuilt.interrupt import HumanInterruptConfig

# Create agent with file operations requiring approval
agent = create_deep_agent(
    tools=[your_tools],
    instructions="Your instructions here",
    interrupt_config={
        "write_file": HumanInterruptConfig(
            allow_ignore=False,
            allow_respond=False,
            allow_edit=False,
            allow_accept=True,
        ),
    }
)
```

## Integrating with MCP Tools

Integrate with tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

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

*   Allow users to customize the full system prompt.
*   Enhance code cleanliness (type hinting, docstrings, formatting).
*   Develop a more robust virtual file system.
*   Create a deep coding agent example.
*   Benchmark the deep research agent example.