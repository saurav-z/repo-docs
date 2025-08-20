# 🧠🤖 DeepAgents: Build Powerful, Deep Agents with Ease

**Unleash the power of AI agents capable of complex tasks by leveraging planning, sub-agents, and file system access.**  [Explore the original repo](https://github.com/hwchase17/deepagents)

DeepAgents provides a Python package that simplifies the creation of sophisticated AI agents, modeled after architectures that enable "deep" task execution. These agents excel where simpler agents fall short, such as in long-term planning and complex actions.

**Key Features:**

*   **Planning:** Built-in planning tool to guide the agent's actions.
*   **Sub-Agents:** Easily integrate and orchestrate sub-agents for specialized tasks and context management.
*   **File System Access:** Simulated file system tools for reading, writing, editing, and listing files within the agent's context.
*   **Customizable Prompts:**  Tailor instructions to guide agent behavior, including a built-in system prompt inspired by Claude Code.
*   **LangGraph Integration:** Agents created are compatible with the LangGraph framework.
*   **MCP Compatibility:** Supports Multi-Server MCP client for enhanced tool integration.
*   **Model Flexibility:** Easily integrates with a wide variety of LangChain models (including custom models and per-subagent model settings).

## Installation

```bash
pip install deepagents
```

## Usage

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

# Agent instructions
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

For a more complex example, see:  [examples/research/research\_agent.py](examples/research/research_agent.py)

## Creating a Custom Deep Agent

The `create_deep_agent` function accepts several parameters:

*   **`tools` (Required):**  A list of functions or LangChain `@tool` objects that the agent will have access to.
*   **`instructions` (Required):**  The instructions or prompt that guide the agent's overall behavior. Combined with the built-in system prompt.
*   **`subagents` (Optional):** Define custom sub-agents, each with their own instructions, tools, and model configurations.
*   **`model` (Optional):**  Specify the language model to use (defaults to `"claude-sonnet-4-20250514"`).  Supports any [LangChain model object](https://python.langchain.com/docs/integrations/chat/) (including custom models via Ollama).  Per-subagent model overrides are also possible.

###  Example:  Using a Custom Model

```python
from deepagents import create_deep_agent
#...
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

### Example:  Per-Subagent Model Override

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

The following features are built into `deepagents` to enable deep task execution:

### System Prompt

A comprehensive, built-in system prompt, inspired by Claude Code, provides instructions for planning, file system use, and sub-agent interaction.

### Planning Tool

A built-in planning tool (similar to Claude Code's TodoWrite) to facilitate task decomposition and strategic planning.

### File System Tools

Four built-in, virtual file system tools (`ls`, `edit_file`, `read_file`, `write_file`) that use LangGraph's State object for isolated file management. Files can be passed in and retrieved using the `"files"` key in the LangGraph state.

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

Supports built-in and custom sub-agents for specialized tasks and improved context management. The `general-purpose` subagent is always available, using the main agent's instructions and tools.

## MCP Integration

`deepagents` integrates with the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters) to support tools via MCP.

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
*   \[ ] Add human-in-the-loop support for tools