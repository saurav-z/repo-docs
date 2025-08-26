# 🧠 Deep Agents: Build Powerful AI Agents with Planning, Subagents, and File System Access

**Create sophisticated AI agents that can tackle complex tasks with `deepagents`, enabling deeper research, coding, and more.**

[Link to Original Repo:](https://github.com/langchain-ai/deepagents)

## Key Features

*   **Deep Planning:** Leverage a built-in planning tool to strategize and execute complex tasks.
*   **Subagent Support:** Easily create and manage subagents for specialized tasks and context isolation.
*   **Virtual File System:** Utilize built-in file system tools for reading, writing, and editing files within the agent's environment.
*   **Customization:** Tailor agents with custom instructions, tools, subagents, and model configurations.
*   **Human-in-the-Loop:** Implement tool execution interrupts for user approval and control.
*   **LangGraph Integration:** Built on LangGraph, offering flexibility for streaming, memory, and other advanced features.
*   **MCP Tools Support:** Run the library with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

## Installation

```bash
pip install deepagents
```

## Usage

(To run the example below, you'll need `pip install tavily-python` and set your Tavily API key in the environment: `export TAVILY_API_KEY="YOUR_TAVILY_API_KEY"`)

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

`create_deep_agent` provides flexibility through these parameters:

### `tools` (Required)

A list of functions or LangChain `@tool` objects available to the agent.

### `instructions` (Required)

The core instructions that guide the agent's behavior, which is combined with a built-in system prompt.

### `subagents` (Optional)

Define custom subagents with their own instructions and tools to handle specific tasks.

```python
from typing import TypedDict, NotRequired, Any
class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    model_settings: NotRequired[dict[str, Any]]

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

Customize the LLM used by the agent. Defaults to `"claude-sonnet-4-20250514"`.  You can pass any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

#### Example: Using a Custom Model

```python
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama

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

## Deep Agent Details

`deepagents` offers built-in components for robust agent creation:

### System Prompt

A comprehensive [built-in system prompt](src/deepagents/prompts.py) inspired by Claude Code, providing detailed guidance for planning, file system interactions, and subagent utilization.

### Planning Tool

A simple, built-in planning tool based on ClaudeCode's TodoWrite, enabling the agent to create and track a plan.

### File System Tools

Four built-in file system tools: `ls`, `edit_file`, `read_file`, `write_file`, providing a virtual file system for agents.  Note that the file system is one level deep.

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

Built-in support for calling subagents, including a `general-purpose` subagent.  You can also specify [custom sub agents](#subagents-optional) with unique instructions and tools for context quarantine and specialized tasks.

### Tool Interrupts

Implement human-in-the-loop approval for tool execution using the `interrupt_config` parameter.

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

## Using with MCP Tools

Integrate `deepagents` with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

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