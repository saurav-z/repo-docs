# 🧠 Deep Agents: Unleash the Power of Advanced AI Agents

**Tired of shallow AI agents that fail on complex tasks?** Deep Agents offers a powerful Python package to create sophisticated AI agents capable of planning, utilizing tools, and handling intricate workflows.  Built on the principles of architectures like "Claude Code", Deep Agents enables you to build intelligent agents that excel at tasks requiring deep reasoning and execution.  [Explore the Deep Agents repository on GitHub](https://github.com/hwchase17/deepagents) for more details.

## Key Features

*   **Flexible Tool Integration:** Easily integrate any function or LangChain `@tool` objects for your agent's capabilities.
*   **Customizable Instructions:** Tailor the agent's behavior with specific instructions to guide its reasoning and actions.
*   **Sub-Agent Support:**  Enable context isolation and specialized task handling with sub-agents. Define their roles, tools, and instructions for efficient task decomposition.
*   **Built-in File System Tools:** Utilize pre-built tools for file manipulation (read, write, list, edit) within a virtual file system.
*   **Human-in-the-Loop:** Incorporate human review and approval for tool executions with comprehensive control over interaction options like accept, edit, and respond.
*   **Async Support:**  Leverage asynchronous tools for increased efficiency, supported by `async_create_deep_agent`.
*   **MCP Compatibility:** Integrated support for MCP tools through the Langchain MCP Adapter library.
*   **Configurable Agent:** Build and deploy fully configurable agents for dynamic control and management.
*   **Built-in Planning:** Includes a basic todo-based planning tool.

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

## Dive Deeper

Explore the [examples/research/research_agent.py](examples/research/research_agent.py) to see a more complex application.  Deep Agents are built using LangGraph, allowing seamless integration with LangGraph's features like streaming, human-in-the-loop, memory, and studio.

## Customization Options

*   **`tools` (Required):**  A list of functions or LangChain `@tool` objects the agent uses.
*   **`instructions` (Required):** The instructions that shape the agent's core behavior and prompt the model for action.
*   **`subagents` (Optional):** Define specialized sub-agents with their own tasks.
*   **`model` (Optional):**  Customize the Large Language Model used by the agent.  Defaults to `claude-sonnet-4-20250514`.
*   **`builtin_tools` (Optional):**  Specify which built-in tools the agent should have access to, providing fine-grained control over capabilities.
*   **`interrupt_config` (Optional):** Customize how human-in-the-loop interrupts work.

## Built-in Components

*   **System Prompt:**  A comprehensive, built-in prompt based on Claude Code's approach, providing detailed instructions and guidance.  This can be further customized.
*   **Planning Tool:** A built-in, Todo-based planning tool to guide task execution.
*   **File System Tools:** A suite of virtual file system tools (ls, edit_file, read_file, write_file) for persistent data storage.
*   **Sub Agents:** Built-in support for calling sub-agents, including a general-purpose subagent and the ability to define custom sub-agents.
*   **Built-in Tools:**  Includes `write_todos`, `write_file`, `read_file`, `ls`, and `edit_file` by default.
*   **Human-in-the-Loop:** Supports manual tool calls via `interrupt_config` allowing you to have fine-grained control over the tool execution

## Roadmap

*   Allow users to customize full system prompt
*   Code cleanliness (type hinting, docstrings, formating)
*   Allow for more of a robust virtual filesystem
*   Create an example of a deep coding agent built on top of this
*   Benchmark the example of [deep research agent](examples/research/research_agent.py)