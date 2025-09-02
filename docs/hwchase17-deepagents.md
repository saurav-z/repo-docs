# 🧠 Deep Agents: Build Powerful, Deeply Reasoning AI Agents

**Unlock the power of sophisticated AI agents capable of planning, executing complex tasks, and interacting with tools through a flexible and extensible framework.** This Python package provides a streamlined approach to building "deep agents" inspired by architectures like Claude Code, enabling you to create AI applications that go beyond simple tool calls.  [Explore the original repo](https://github.com/hwchase17/deepagents).

**Key Features:**

*   **Built-in Planning Tool:**  Facilitates strategic task decomposition and execution.
*   **Sub-Agent Architecture:**  Enables context quarantine and specialized reasoning through nested agents.
*   **Virtual File System Tools:**  Provides `ls`, `edit_file`, `read_file`, and `write_file` to simulate file interactions, enhancing agent capabilities.
*   **Human-in-the-Loop Support:** Incorporates human oversight and interaction for tool execution with `interrupt_config`.
*   **Modular & Extensible:**  Easily integrate custom tools, instructions, and sub-agents to tailor agents to your specific needs.
*   **Async Support:** Provides asynchronous capabilities to leverage async tools and services
*   **MCP Integration:** Integrate with MCP tools to expand agent capabilities
*   **Configurable Agent:** Build agents from a config passed in.

## Installation

```bash
pip install deepagents
```

## Core Components & Customization

*   **Tools:**  Pass in a list of functions or LangChain `@tool` objects the agent can utilize.
*   **Instructions:**  Craft specific prompts to guide the agent's behavior and reasoning.
*   **Subagents (Optional):**  Define specialized sub-agents with custom instructions and tool access.  Useful for complex tasks and context management.
*   **Model (Optional):**  Customize the LLM used by the agent by specifying a LangChain model object.
*   **Built-in Tools (Optional):** Control access to the default set of tools, which include filesystem tools and tools for writing todos.

## Example Usage

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

## Advanced Features

*   **Deep Agent Details:**  The core of `deepagents` utilizes a powerful built-in system prompt, planning tools, file system tools, sub-agents, and human-in-the-loop capabilities to create agents capable of completing more complex tasks.
*   **Human-in-the-Loop:**  The `interrupt_config` allows for human oversight and interaction for tool execution.
*   **Asynchronous Agent Creation:**  Leverage asynchronous operations using `async_create_deep_agent` for improved efficiency and compatibility with async tools.
*   **MCP Integration:** Create agents to leverage MCP tools using the  [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).
*   **Configurable Agent:** Create agents from a config passed in.

## Roadmap

*   [ ] Allow users to customize full system prompt
*   [ ] Code cleanliness (type hinting, docstrings, formating)
*   [ ] Allow for more of a robust virtual filesystem
*   [ ] Create an example of a deep coding agent built on top of this
*   [ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)