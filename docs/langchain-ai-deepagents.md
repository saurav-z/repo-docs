# Deep Agents: Build Powerful, Autonomous LLM Applications

**Unleash the potential of advanced LLMs with `deepagents`, a Python package that empowers you to create intelligent agents capable of planning, acting, and executing complex tasks.** ([Original Repo](https://github.com/langchain-ai/deepagents))

## Key Features:

*   **Modular Architecture:**  Easily build agents by combining a planning tool, sub-agents, a virtual file system, and detailed prompts.
*   **Built-in Planning Tool:**  A simple planning tool (inspired by Claude Code) to structure agent actions.
*   **Virtual File System:**  Mock file system tools (`ls`, `edit_file`, `read_file`, `write_file`) for agent interaction and state management (single directory depth).
*   **Sub-Agent Support:**  Facilitates modular design and "context quarantine" with the ability to define and utilize custom sub-agents.
*   **Customization:** Easily tailor your agents with custom tools, instructions, and sub-agents, while defining the model to use.
*   **Human-in-the-Loop:** Integrates human approval for specific tool executions to prevent unintended actions.
*   **LangGraph Integration:**  `deepagents` agents are built on LangGraph, offering seamless integration with other LangGraph features like streaming, memory, and studio.
*   **MCP Compatibility:** Run `deepagents` with MCP tools, by using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

## Installation

```bash
pip install deepagents
```

## Quickstart: Research Agent Example

**(Requires `pip install tavily-python`)**

This example demonstrates how to create a simple research agent using the Tavily search tool.

```python
import os
from typing import Literal

from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Define a search tool.
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

# Create a custom research instruction for the agent
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

For a more complex example, see [examples/research/research\_agent.py](examples/research/research_agent.py).

## Creating Custom Deep Agents

The `create_deep_agent` function is the core of `deepagents`.  You can customize agents by configuring the following parameters:

*   **`tools` (Required):** A list of functions or LangChain `@tool` objects the agent can use.
*   **`instructions` (Required):** A custom prompt that acts as part of the agent's instructions.
*   **`subagents` (Optional):**  Define sub-agents with their own instructions and tools for complex task decomposition.
*   **`model` (Optional):** Specify a custom LangChain model for the agent.  Defaults to `"claude-sonnet-4-20250514"`.
*   **`builtin_tools` (Optional):** Override which default built-in tools are used.

### Example: Using a Custom Model

```python
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama

model = ChatOllama(model="gpt-oss:20b")
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
    ...
)
```

### Example: Per-subagent model override

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

`deepagents` leverages several built-in components to create a powerful agent:

*   **System Prompt:** Includes a detailed system prompt inspired by Claude Code, containing instructions for built-in tools and sub-agents (customizable through `instructions`).
*   **Planning Tool:** The agent can create a plan using the built in planning tool.
*   **File System Tools:** Basic file system tools (`ls`, `edit_file`, `read_file`, `write_file`) for virtual file management (single directory depth).
*   **Sub Agents:** Calls sub-agents (including a default `general-purpose` subagent) to enhance modularity and context management.
*   **Built-in Tools:** Agents include a set of built in tools: `write_todos`, `write_file`, `read_file`, `ls`, `edit_file`. These can be disabled.

### Human-in-the-Loop

Support for human-in-the-loop approval for tool execution via the `interrupt_config` parameter.

*   **HumanInterruptConfig:**
    *   `allow_respond`: Whether the user can add a text response
    *   `allow_edit`: Whether the user can edit the tool arguments
    *   `allow_accept`: Whether the user can accept the tool call

### MCP Tools

`deepagents` integrates with MCP tools (via the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters)).

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
*   Code cleanliness (type hinting, docstrings, formatting)
*   Expand virtual file system capabilities.
*   Develop a deep coding agent example.
*   Benchmark the research agent example.