# Deep Agents: Build Advanced AI Agents with Ease

**Unlock the power of sophisticated AI agents that plan, research, and execute complex tasks with `deepagents`, a Python package built for creating powerful, flexible, and customizable AI agents.**  [Explore the Source Code on GitHub](https://github.com/hwchase17/deepagents)

<img src="deep_agents.png" alt="deep agent" width="600"/>

`deepagents` provides a streamlined approach to building "deep" agents, inspired by architectures like Claude Code, by combining planning, sub-agents, file system access, and detailed prompts to achieve advanced task execution.

## Key Features

*   **Simplified Agent Creation:** Easily build deep agents with a single function call.
*   **Planning and Execution:** Built-in planning tools and virtual file system to manage tasks and data.
*   **Sub-Agents:** Supports nested agents with custom instructions and tool access for context management and specialized tasks.
*   **Customization:** Easily integrate custom tools, prompts, and models to tailor agents to specific needs.
*   **LangGraph Integration:**  Built on LangGraph, enabling integration with streaming, human-in-the-loop, memory, and other LangGraph features.
*   **MCP Integration:** Works with Langchain's MCP tools for enhanced functionality.

## Installation

```bash
pip install deepagents
```

## Quickstart

(Requires `pip install tavily-python`)

```python
import os
from typing import Literal

from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Define your research tool
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

# Craft your instructions
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# Build your agent
agent = create_deep_agent(
    [internet_search],
    research_instructions,
)

# Run the agent
result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
```

## Customizing Your Deep Agent

### Tools

*   Provide a list of functions or LangChain `@tool` objects for the agent to utilize.
*   These tools enable the agent to interact with the external environment and perform actions.

### Instructions

*   Craft detailed instructions to guide the agent's behavior and provide context.
*   These instructions are integrated into the agent's prompt.

### Subagents

*   Define subagents with unique roles, instructions, and toolsets for specialized tasks.
*   Subagents help in context management and task decomposition, improving agent performance.
*   Specify subagents as a list of dictionaries, each following the schema defined in the original README.

### Model

*   Customize the language model used by the agent.
*   By default, it uses `"claude-sonnet-4-20250514"`. You can change this by passing any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).
*   Example:

```python
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")
agent = create_deep_agent(tools=tools, instructions=instructions, model=model)
```

### Built-in Tools

*   Control the built-in tools available to the agent.
*   Choose which tools to include (e.g., file system, todo list) or exclude specific functionalities.
*   By default, five built-in tools are included, and can be disabled via the `builtin_tools` parameter.

### Tool Interrupts

*   Configure human-in-the-loop approval for tool execution.
*   Set up approvals, rejections, and more.
*   Utilize the `HumanInterruptConfig` class.

## Deep Agent Components

`deepagents` offers several built-in components to enhance agent capabilities:

### System Prompt

*   A comprehensive system prompt inspired by Claude Code, providing detailed instructions on using planning, file system tools, and sub-agents.
*   This detailed prompt is essential for creating a "deep" agent.

### Planning Tool

*   A built-in planning tool, based on ClaudeCode's TodoWrite tool, to help the agent devise a plan of action.

### File System Tools

*   Virtual file system tools: `ls`, `edit_file`, `read_file`, and `write_file`.
*   These tools enable the agent to manage and interact with a simulated file system using the LangGraph's State object.

### Sub Agents

*   The built-in ability to call sub agents, based on Claude Code.
*   Sub agents are available to the main agent at all times.
*   Support for custom sub agents with unique roles and toolsets.

### Built-in Tools

*   Five default tools are included: `write_todos`, `write_file`, `read_file`, `ls`, and `edit_file`.

## Integration with MCP

*   Integrate `deepagents` with [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters) to incorporate MCP tools.

## Roadmap

*   Allow users to customize full system prompt
*   Code cleanliness (type hinting, docstrings, formating)
*   Allow for more of a robust virtual filesystem
*   Create an example of a deep coding agent built on top of this
*   Benchmark the example of [deep research agent](examples/research/research_agent.py)