# Deep Agents: Build Advanced, Autonomous AI Agents with Ease

**Unlock the power of intelligent agents capable of complex tasks with the `deepagents` Python package.**

[View the Project on GitHub](https://github.com/langchain-ai/deepagents)

`deepagents` empowers you to create sophisticated AI agents that can plan, execute, and adapt, drawing inspiration from leading projects like Claude Code. This library provides a streamlined approach to building deep agents, incorporating key components like planning, sub-agents, and a virtual file system, allowing your AI to excel at intricate and extended tasks.

## Key Features

*   **Planning & Execution:** Built-in planning tools enable your agent to break down complex problems into manageable steps.
*   **Sub-Agents:** Leverage sub-agents for context quarantine and specialized tasks, enhancing overall performance.
*   **Virtual File System:** Access a virtual file system with `ls`, `edit_file`, `read_file`, and `write_file` to manage and manipulate data.
*   **Customization:** Easily configure your agent with custom tools, instructions, sub-agents, and model settings.
*   **Built-in Tools:** Leverage ready-to-use tools for file system operations, research, and more.
*   **Human-in-the-Loop:** Integrate human approval for tool execution with interrupt configuration for specific tools.
*   **MCP Integration:** Integrate with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

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

The agent created with `create_deep_agent` is just a LangGraph graph - so you can interact with it (streaming, human-in-the-loop, memory, studio) in the same way you would any LangGraph agent.

## Creating a Custom Deep Agent

`create_deep_agent` offers flexibility through the following parameters:

### `tools` (Required)

Provide a list of functions or LangChain `@tool` objects that your agent (and sub-agents) can utilize.

### `instructions` (Required)

Define the core prompt to guide your deep agent's behavior. This is combined with a built-in system prompt.

### `subagents` (Optional)

Specify custom sub-agents, each with unique instructions and tool access.

```python
class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    model_settings: NotRequired[dict[str, Any]]
```

*   **name:** Subagent name.
*   **description:** Description for the main agent.
*   **prompt:** The prompt for the subagent.
*   **tools:** List of tools accessible to the subagent.
*   **model_settings:** Optional model configuration.

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

Customize the LLM used by the agent.  Defaults to "claude-sonnet-4-20250514".

#### Example: Using a Custom Model

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

Control the availability of built-in tools (e.g., `write_todos`, file system tools).

```python
# Only give agent access to todo tool, none of the filesystem tools
builtin_tools = ["write_todos"]
agent = create_deep_agent(..., builtin_tools=builtin_tools, ...)
```

## Deep Agent Components

`deepagents` incorporates several key components for building effective deep agents:

### System Prompt

A comprehensive built-in system prompt, inspired by Claude Code, provides instructions for using tools and sub-agents.

### Planning Tool

A basic built-in planning tool (inspired by ClaudeCode's TodoWrite tool) to help the agent create a plan.

### File System Tools

A set of virtual file system tools: `ls`, `edit_file`, `read_file`, and `write_file`.

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

The ability to call sub-agents. There's a built in `general-purpose` subagent and you can also define [custom sub agents](#subagents-optional) with their own instructions and tools.

### Built-in Tools

Five built-in tools are provided:

-   `write_todos`
-   `write_file`
-   `read_file`
-   `ls`
-   `edit_file`

### Tool Interrupts

Enable human-in-the-loop approval for tool execution with `interrupt_config`:

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