# 🧠🤖 Deep Agents: Build Powerful, "Deep" LLM Agents with Ease

**Unleash the power of LLMs for complex tasks with Deep Agents, a Python package designed to create agents capable of planning, utilizing tools, and managing sub-agents for advanced problem-solving.**  [See the original repository](https://github.com/langchain-ai/deepagents)

Deep Agents moves beyond basic agent architectures by implementing a combination of planning, sub-agents, file system access, and detailed prompts to tackle challenging projects.

<img src="deep_agents.png" alt="deep agent" width="600"/>

## Key Features:

*   **Effortless Agent Creation:** Easily create deep agents using the `create_deep_agent` function.
*   **Planning Tool:** Built-in planning tool inspired by Claude Code.
*   **Virtual File System:** Integrated file system tools (`ls`, `edit_file`, `read_file`, `write_file`) for managing and manipulating data.
*   **Sub-Agent Support:** Enables context isolation and modularity through the use of sub-agents, allowing for specialized tasks and enhanced performance.
*   **Customizable Prompts:** Customize the agent's instructions and system prompt for specific use cases.
*   **Tool Interrupts:** Implement human-in-the-loop approval for tool execution.
*   **MCP Compatibility:** Integrates seamlessly with Multi-Server MCP Clients for extended capabilities.

## Installation

```bash
pip install deepagents
```

## Getting Started

Here's a simple example to demonstrate how to use Deep Agents with the Tavily search tool:

**(Requires `pip install tavily-python` and a Tavily API key)**

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

For a more complex example, refer to [examples/research/research_agent.py](examples/research/research_agent.py).  The agent created with `create_deep_agent` is a LangGraph graph, enabling seamless integration with LangGraph features.

## Core Components

### `create_deep_agent` Parameters

*   **`tools` (Required):**  A list of functions or LangChain `@tool` objects that the agent can utilize.
*   **`instructions` (Required):**  Custom instructions to guide the agent's behavior.
*   **`subagents` (Optional):** Define custom sub-agents with specific instructions and tools.
*   **`model` (Optional):**  Specify a custom LangChain model (defaults to `"claude-sonnet-4-20250514"`).
*   **`builtin_tools` (Optional):** Control which built-in tools the agent has access to.

### Deep Agent Details

*   **System Prompt:** A comprehensive, built-in system prompt inspired by Claude Code, designed to guide the agent's behavior.
*   **Planning Tool:** A basic built-in planning tool.
*   **File System Tools:**  Virtual file system tools (`ls`, `edit_file`, `read_file`, `write_file`) that use LangGraph's State object for file management.
*   **Sub Agents:** Ability to call subagents, including a built-in general-purpose subagent, for context isolation and specialized tasks.
*   **Built-In Tools:** Deep Agents include: `write_todos`, `write_file`, `read_file`, `ls`, `edit_file`.
*   **Tool Interrupts:** Allows for human-in-the-loop approval for tool execution.

### Custom Model Examples

#### Using Ollama:

```python
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gpt-oss:20b",
)
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
)
```

#### Per-Subagent Model Override:

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

## MCP Integration

Integrate Deep Agents with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

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