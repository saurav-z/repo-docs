# Deep Agents: Build Advanced, Autonomous LLM Agents with Python

**Unlock the power of advanced LLM agents with `deepagents`, a Python library for creating sophisticated, multi-step agents capable of complex task execution.**  This library enables you to build agents that can plan, use tools, and interact with their environment to solve intricate problems.

[View the original repository on GitHub](https://github.com/langchain-ai/deepagents)

## Key Features

*   **Planning and Execution:** Leverages a built-in planning tool to enable agents to create and execute multi-step plans.
*   **Sub-Agent Architecture:** Supports the creation and integration of sub-agents for modularity, context management, and specialized tasks.
*   **File System Tools:** Provides built-in tools for file manipulation (read, write, edit, list) within a sandboxed environment.
*   **Customizable Prompting:** Offers a built-in system prompt that is heavily inspired by Claude Code's prompt and allows for customization.
*   **Tool Interrupts:** Supports human-in-the-loop approval for tool execution, allowing for control and oversight.
*   **MCP Integration:** Compatible with the Langchain MCP Adapter library, allowing agents to utilize MCP tools for advanced capabilities.
*   **Model Flexibility:** Supports custom LLM models through the `model` parameter, providing flexibility in choosing your preferred LLM.
*   **Custom Sub Agent Model Settings**: Allows to override the underlying LLM model settings on a per-subagent level.

## Installation

```bash
pip install deepagents
```

## Usage

Here's a basic example demonstrating how to create a research agent that uses the Tavily search tool:

**(Note: Requires `pip install tavily-python` and a Tavily API key.)**

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
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# Instructions
research_instructions = """You are an expert researcher.  Conduct thorough research, and then write a polished report.
You have access to the internet search tool"""

# Create the agent
agent = create_deep_agent(
    [internet_search],
    research_instructions,
)

# Invoke the agent
result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
```

For a more advanced example, see `examples/research/research_agent.py`.

## Creating a Custom Deep Agent

The `create_deep_agent` function accepts the following parameters:

### `tools` (Required)

A list of functions or LangChain `@tool` objects that the agent and subagents can use.

### `instructions` (Required)

Instructions to be included in the agent's prompt.

### `subagents` (Optional)

A list of dictionaries defining custom sub-agents.  Each sub-agent definition includes:

*   `name`: Sub-agent's name.
*   `description`: Sub-agent's description.
*   `prompt`: Sub-agent's prompt.
*   `tools` (Optional): List of tools for the subagent. Defaults to all tools.
*   `model_settings` (Optional): Override the model settings for the subagent (e.g., different LLM, temperature).

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

Specify a custom LangChain model object to use, defaulting to `"claude-sonnet-4-20250514"`. This lets you use models from other providers.
Example:

```python
from deepagents import create_deep_agent
from langchain_ollama.chat_models import ChatOllama

model = ChatOllama(model="gpt-oss:20b")

agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
    ...
)
```

#### Per-subagent model override (optional)

Use a fast, deterministic model for a critique sub-agent, while keeping a different default model for the main agent and others:

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

## Deep Agent Components

`deepagents` provides key components for building effective deep agents:

### System Prompt

A comprehensive built-in system prompt, inspired by Claude Code, to guide agent behavior, including planning, tool use, and sub-agent interaction.

### Planning Tool

A built-in tool (based on ClaudeCode's TodoWrite) to facilitate agent planning.

### File System Tools

Four built-in file system tools (`ls`, `edit_file`, `read_file`, `write_file`) that simulate a file system using LangGraph's State object.

```python
agent = create_deep_agent(...)

result = agent.invoke({
    "messages": ...,
    # Pass in files to the agent using this key
    # "files": {"foo.txt": "foo", ...}
})

# Access any files afterwards like this
result["files"]
```

### Sub Agents

Built-in support for sub-agents (similar to Claude Code) for context management and task specialization. Includes a `general-purpose` subagent with all tools.  You can also define [custom sub agents](#subagents-optional).

### Tool Interrupts

Implement human-in-the-loop approval for tool execution using the `interrupt_config` parameter. Control whether users can skip, respond, edit, or accept tool calls.

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

## MCP Integration

Integrate with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

**(Requires `pip install langchain-mcp-adapters`)**

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