# Deep Agents: Build Powerful LLM Agents with Planning, Tools, and Subagents

**Unlock the potential of complex tasks with `deepagents`, a Python package that empowers you to create sophisticated, multi-step LLM agents.**

[![GitHub Repo stars](https://img.shields.io/github/stars/langchain-ai/deepagents?style=social)](https://github.com/langchain-ai/deepagents)

`deepagents` provides a general-purpose framework inspired by advancements like Claude Code, allowing you to build LLM agents capable of planning, utilizing tools, managing a virtual file system, and leveraging subagents for enhanced performance. This enables your agents to tackle intricate problems beyond the capabilities of simple, single-step agents.

**Key Features:**

*   **Planning Tool:** Built-in planning functionality to guide the agent's actions.
*   **Subagents:** Easily integrate subagents with custom instructions and tools for specialized tasks and improved context management.
*   **Virtual File System:**  Mock file system using LangGraph's State object, allowing for file operations within the agent's context.
*   **Tool Interrupts:**  Human-in-the-loop approval for tool execution, enhancing control and safety.
*   **MCP Compatibility**: Integrate with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters)
*   **Customizable:** Configure tools, instructions, and subagents to tailor agents for specific applications.

## Installation

```bash
pip install deepagents
```

## Usage

**(Requires `pip install tavily-python` for the example)**

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

Customize your deep agents using the `create_deep_agent` function with these parameters:

### `tools` (Required)

*   A list of functions or LangChain `@tool` objects that the agent and its subagents can utilize.

### `instructions` (Required)

*   A string that provides instructions to the main agent, shaping its behavior and goals. This complements a built-in system prompt.

### `subagents` (Optional)

*   Define custom subagents with their own names, descriptions, prompts, and tools to handle specific tasks.

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

*   Specify a LangChain model object to customize the LLM used by the agent. By default, uses `"claude-sonnet-4-20250514"`.

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

**Per-subagent model override (optional)**

Override the default model for specific sub-agents.

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

`deepagents` incorporates the following features to facilitate deep task completion:

### System Prompt

*   A comprehensive, built-in system prompt, inspired by Claude Code, that provides detailed instructions for planning, using tools, and interacting with subagents. The default system prompt can be customized.

### Planning Tool

*   A built-in planning tool based on ClaudeCode's TodoWrite tool, helping the agent create and follow a plan.

### File System Tools

*   Four built-in file system tools (`ls`, `edit_file`, `read_file`, `write_file`) that mock file operations using LangGraph's State object.

### Sub Agents

*   Built-in support for calling subagents, including a general-purpose subagent.  You can also create [custom sub agents](#subagents-optional) to partition tasks.

### Tool Interrupts

*   Control tool execution with human-in-the-loop approval, enhancing safety and oversight.  Configure tool-specific interrupt settings using `HumanInterruptConfig` .

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

Integrate with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters) for enhanced tool access.

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

*   Allow users to customize the full system prompt
*   Code cleanliness (type hinting, docstrings, formating)
*   Allow for more of a robust virtual filesystem
*   Create an example of a deep coding agent built on top of this
*   Benchmark the example of [deep research agent](examples/research/research_agent.py)

[View the original repository](https://github.com/langchain-ai/deepagents)