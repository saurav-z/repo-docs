# Deep Agents: Build Powerful LLM-Powered Agents for Complex Tasks

**Unlock the potential of LLMs to perform complex tasks with Deep Agents, a Python package designed to build sophisticated, multi-step agents.**  Check out the original repo: [https://github.com/hwchase17/deepagents](https://github.com/hwchase17/deepagents)

## Key Features

*   **Planning and Execution:** Enables agents to plan and execute multi-step tasks, moving beyond simple, single-call interactions.
*   **Modular Architecture:**  Leverages a combination of planning tools, sub-agents, a virtual file system, and detailed prompting to create robust agents.
*   **Customization:** Easily tailor agents to your specific needs by defining tools, instructions, and sub-agents.
*   **Built-in Tools:** Provides essential tools for planning, file manipulation (read, write, list, edit), and a built-in general-purpose subagent, out-of-the-box.
*   **Human-in-the-Loop Support:**  Integrates human approval for tool execution, allowing for safe and controlled operation.
*   **MCP Integration:** Runs with MCP tools.

## Installation

```bash
pip install deepagents
```

## Core Concepts

Deep Agents are built upon these key components:

*   **System Prompt:**  A comprehensive prompt designed to guide the agent's behavior, inspired by Claude Code and made more general-purpose.
*   **Planning Tool:**  A built-in tool for generating and managing task plans.
*   **File System Tools:**  Virtual file system tools (ls, edit\_file, read\_file, write\_file) for managing persistent, or temporary, data within an agent's operation.
*   **Sub Agents:** Allows breaking down large tasks into sub-tasks, handled by separate agents with custom instructions and tool access.
*   **Tool Interrupts:**  Configurable human-in-the-loop approval for tool executions, ensuring control and safety.

## Usage

Here's a basic example of creating a Deep Agent with a search tool:

**(Requires `pip install tavily-python` for this example)**

```python
import os
from typing import Literal

from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Search tool
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

# Instructions for research agent
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

For a more advanced example, see `examples/research/research_agent.py`.

The `create_deep_agent` function returns a LangGraph graph, allowing you to integrate it with other LangGraph features (streaming, memory, etc.).

## Customizing Your Deep Agent

Customize your deep agents with these parameters to `create_deep_agent`:

### `tools` (Required)

A list of functions or LangChain `@tool` objects the agent (and sub-agents) can use.

### `instructions` (Required)

Custom instructions for the agent.  These are combined with a built-in system prompt.

### `subagents` (Optional)

Define custom sub-agents with their own instructions, tools, and model configurations.

**Example:**
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

Specify a custom LangChain model object. Defaults to `"claude-sonnet-4-20250514"`.

**Example:**

```python
from deepagents import create_deep_agent
from langchain_ollama import init_chat_model

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

### `builtin_tools` (Optional)

Control which built-in tools the agent has access to.

**Example:**

```python
builtin_tools = ["write_todos"]  # Only allow the todo tool
agent = create_deep_agent(..., builtin_tools=builtin_tools, ...)
```

### Per-subagent model override (optional)

Override the default model for specific sub-agents.

**Example:**

```python
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

### Human Interrupt Configuration (Optional)

Configure which tools require human approval before execution.

**Example:**

```python
from deepagents import create_deep_agent
from langgraph.prebuilt.interrupt import HumanInterruptConfig

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

## MCP Integration

Integrate with MCP tools by using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

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

*   Allow users to customize full system prompt
*   Code cleanliness (type hinting, docstrings, formatting)
*   Allow for more of a robust virtual filesystem
*   Create an example of a deep coding agent built on top of this
*   Benchmark the example of [deep research agent](examples/research/research_agent.py)