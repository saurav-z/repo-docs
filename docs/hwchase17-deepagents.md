# 🧠 Deep Agents: Build Powerful, Intelligent Agents with Ease

**Create advanced agents capable of complex reasoning and problem-solving with the `deepagents` Python package.** 

[View the original repo on GitHub](https://github.com/hwchase17/deepagents)

`deepagents` provides a flexible framework for constructing "deep agents" that can tackle intricate tasks through planning, sub-agents, and access to tools. It empowers developers to move beyond simple agent architectures, enabling more sophisticated and effective AI solutions.

## Key Features

*   **Simplified Deep Agent Creation:** Easily build agents using pre-built components and customizable configurations.
*   **Planning Tool:** Includes a built-in planning tool to guide the agent's actions and maintain focus on long-term goals.
*   **Virtual File System:** Offers a mock file system for agents to manage files and data without affecting the underlying system.
*   **Sub-Agents:** Supports the creation and integration of sub-agents for specialized tasks and context isolation.
*   **Human-in-the-Loop:** Integrates human approval for tool execution through the `interrupt_config` parameter.
*   **Built-in Tools:** Includes a default set of tools. `write_todos`, `write_file`, `read_file`, `ls`, and `edit_file`.
*   **Asynchronous Support:**  Provides `async_create_deep_agent` for use with asynchronous tools.
*   **MCP Integration:** Supports the use of MCP tools with the Langchain MCP Adapter.
*   **Configurable Agents**: Allows you to create agents that can be built via a config passed in.

## Installation

```bash
pip install deepagents
```

## Usage Examples

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

**See the full example at [examples/research/research_agent.py](examples/research/research_agent.py)**.

## Customization Options

*   **Tools:** Provide a list of functions or LangChain `@tool` objects.
*   **Instructions:** Customize the prompt to guide agent behavior.
*   **Sub-Agents:** Define sub-agents for specialized tasks.
*   **Model:** Specify a LangChain model object.
*   **Built-in Tools:** Control which built-in tools are accessible.

## Further Details

*   **Deep Agent Components**: Learn more about System prompts, Planning Tools, File System Tools, Sub Agents, Human-in-the-Loop and Built-In Tools in the documentation.

## Roadmap

*   Allow users to customize full system prompt
*   Code cleanliness (type hinting, docstrings, formatting)
*   Allow for more of a robust virtual filesystem
*   Create an example of a deep coding agent built on top of this
*   Benchmark the example of [deep research agent](examples/research/research_agent.py)