# DeepAgents: Build Powerful LLM Agents for Complex Tasks

**Unlock the power of LLMs to tackle complex challenges with DeepAgents, a Python package designed for building advanced, self-improving agents.**  [Explore the original repo](https://github.com/hwchase17/deepagents)

DeepAgents enables the creation of sophisticated agents that can plan, utilize tools, and solve problems beyond the capabilities of simpler LLM-based systems. Inspired by cutting-edge applications like Claude Code, DeepAgents provides a streamlined and versatile approach to agent development.

## Key Features

*   **Planning Tool:** Built-in planning to break down complex tasks.
*   **Sub-Agents:** Create and manage specialized sub-agents for modular problem-solving and context management.
*   **Virtual File System:** Mock file system tools (ls, read/write, edit) for data manipulation and persistent memory within the agent's context.
*   **Customizable System Prompt:**  Fine-tune the agent's behavior with an adaptable system prompt.
*   **Human-in-the-Loop Support:** Integrate human oversight and control for tool execution with configurable interrupt options.
*   **Asynchronous Support:** Built-in support for asynchronous tools with `async_create_deep_agent`.
*   **LangChain MCP Adapter:**  Integrate with the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters) for asynchronous MCP tools.
*   **Configurable Agents:** Dynamically configure agents using JSON configs with `create_configurable_agent`.
*   **Model Customization:** Use any [LangChain model object](https://python.langchain.com/docs/integrations/chat/) for your agent.

## Installation

```bash
pip install deepagents
```

## Usage Example

**(Requires `pip install tavily-python`)**

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

# Instructions
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.
"""

# Create and invoke the agent
agent = create_deep_agent([internet_search], research_instructions)
result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
```

Refer to  [examples/research/research_agent.py](examples/research/research_agent.py) for a more advanced demonstration.

## Further Customization

*   **Tools:** Provide a list of functions or LangChain `@tool` objects for the agent to use.
*   **Instructions:**  Customize the agent's persona and behavior with detailed instructions.
*   **Subagents:**  Define sub-agents with specialized prompts and tools to handle specific subtasks.
*   **Model:** Specify any LangChain chat model.
*   **Built-in Tools:**  Control access to pre-built tools.

## Deep Agent Details

DeepAgents incorporates several components to enable advanced agent capabilities:

*   **System Prompt:**  A comprehensive prompt guides the agent's decision-making and tool usage (customizable).
*   **Planning Tool:** A built-in tool facilitates task planning and organization.
*   **File System Tools:** Simulated file system tools provide persistent memory and allow for saving and retrieving data between agent iterations.
*   **Sub Agents:** Utilize sub-agents for specialized tasks, improved context management, and modularity.
*   **Built-in Tools:** Access a core set of tools like `write_todos`, `write_file`, `read_file`, `ls`, and `edit_file`.
*   **Human-in-the-Loop:** Integrate human feedback to allow for approval, editing, and response for tool calls.

## Async

For async tools, use `from deepagents import async_create_deep_agent`.

## MCP

Integrate with the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

## Configurable Agent

Dynamically configure agents using JSON configs with `create_configurable_agent`.

## Roadmap

*   [ ] Allow users to customize full system prompt
*   [ ] Code cleanliness (type hinting, docstrings, formating)
*   [ ] Allow for more of a robust virtual filesystem
*   [ ] Create an example of a deep coding agent built on top of this
*   [ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)