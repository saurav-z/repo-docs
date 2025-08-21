# 🧠 Deep Agents: Build Intelligent, Multi-Step AI Agents with Ease

**Deep Agents empowers you to create advanced AI agents capable of complex tasks through planning, sub-agents, and a virtual file system, enabling deeper reasoning and more sophisticated problem-solving.** [View the original repo](https://github.com/langchain-ai/deepagents)

<img src="deep_agents.png" alt="deep agent" width="600"/>

Inspired by the architecture of cutting-edge agents like Claude Code, Deep Agents provides a general-purpose Python package to build agents that go beyond simple tool calls. By combining a planning tool, sub-agents, a virtual file system, and detailed prompts, Deep Agents allows you to tackle intricate research, coding, and other multi-step applications with ease.

## Key Features

*   **Planning Tool:** Built-in planning tool to guide agents through complex workflows.
*   **Sub-Agents:** Utilize specialized sub-agents for context quarantine and custom instructions.
*   **Virtual File System:** Mock file system for secure and isolated file operations.
*   **Customizable Prompts:**  Tailor agent behavior with custom instructions and model parameters.
*   **LangGraph Integration:** Built on top of LangGraph, making it easy to integrate with other LangChain components.
*   **MCP Compatibility:** Supports Multi-Server MCP Client integrations

## Installation

```bash
pip install deepagents
```

## Usage Examples

### Simple Research Agent

(Requires `pip install tavily-python`)

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

# Instructions for the agent
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# Create and invoke the agent
agent = create_deep_agent(
    [internet_search],
    research_instructions,
)
result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
```

For a more advanced example, see the [research agent](examples/research/research_agent.py).

###  Customizing your Deep Agent

Use `create_deep_agent` with these parameters:

### `tools` (Required)

A list of functions or LangChain `@tool` objects that the agent can use.

### `instructions` (Required)

A string that provides the agent's role and task instructions. Combined with a built-in system prompt.

### `subagents` (Optional)

Define custom sub-agents with their own instructions and tools for specialized tasks.

```python
from deepagents import create_deep_agent
from typing import Any, NotRequired, TypedDict

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

By default, `deepagents` uses `"claude-sonnet-4-20250514"`. You can customize the model by passing any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

#### Using a Custom Model (e.g., Ollama)

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

#### Per-Subagent Model Override

Override default model for sub-agents.

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

### System Prompt

`deepagents` includes a built-in system prompt, inspired by Claude Code.  It guides agents on tool usage, planning, file management, and sub-agent calls.

### Planning Tool

A simple planning tool, based on ClaudeCode's TodoWrite, for creating and managing execution plans.

### File System Tools

Four built-in tools: `ls`, `edit_file`, `read_file`, and `write_file`, utilizing a virtual file system managed by LangGraph's State object.

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

Built-in support for sub-agents, including a general-purpose agent and the ability to define custom sub-agents.

## MCP Integration

Integrate `deepagents` with the Langchain MCP Adapter Library.

(Requires `pip install langchain-mcp-adapters`)

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
*   [ ] Add human-in-the-loop support for tools