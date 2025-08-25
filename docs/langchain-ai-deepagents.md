# DeepAgents: Build Powerful, Intelligent Agents with Ease

**Unlock the power of complex task automation with DeepAgents, a Python package for creating sophisticated, multi-step AI agents.**  [View the GitHub Repository](https://github.com/langchain-ai/deepagents)

DeepAgents empowers you to build AI agents capable of planning, utilizing tools, and interacting with files, going beyond simple LLM-based agents.  Inspired by advancements like Claude Code, DeepAgents provides a streamlined way to create agents for advanced applications such as deep research, code generation, and more.

## Key Features

*   **Simplified Deep Agent Creation:** Easily construct agents with planning capabilities, sub-agents, and file system interaction.
*   **Built-in Planning & File System Tools:** Leverage pre-built planning tools and a mock file system for straightforward task execution.
*   **Customizable Sub-Agents:** Define and integrate specialized sub-agents to handle specific tasks and enhance context management.
*   **Human-in-the-Loop Interrupts:** Integrate approval steps for crucial tool executions, with customizable prefixes.
*   **Modular & Extensible:** Built on LangGraph, enabling seamless integration with LangChain's ecosystem, including streaming, memory, and other features.
*   **MCP Tool Integration:**  Leverage agents with Multi-Server MCP tools by using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

## Installation

```bash
pip install deepagents
```

## Usage & Examples

**1. Basic Research Agent**

Create a research agent capable of internet searches:

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

**(Requires `pip install tavily-python`)**

**2.  Advanced Example:**

Explore a more complex example in `examples/research/research_agent.py` for a deeper understanding.

## Customizing Your Deep Agent

`create_deep_agent` accepts parameters for customization:

### `tools` (Required)

Pass a list of functions or LangChain `@tool` objects that your agent will utilize.

### `instructions` (Required)

Provide instructions to steer the agent's behavior.  A [built-in system prompt](src/deepagents/prompts.py) is also included.

### `subagents` (Optional)

Define and incorporate custom sub-agents for modular task management:

```python
from typing import TypedDict, NotRequired, Any

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

Customize the LLM model.  Defaults to `"claude-sonnet-4-20250514"`.  Use any LangChain model object for model flexibility.
```python
from deepagents import create_deep_agent
from langchain_ollama import Ollama

model = Ollama(model="gpt-oss:20b")
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
    ...
)
```

**Per-Subagent Model Override:** Refine agent behavior by assigning a specific model to each sub-agent.

```python
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

## Deep Agent Core Components

*   **System Prompt:** A foundational prompt based on Claude Code for orchestrating agent actions. The prompt includes instructions for the planning tool, file system tools, and sub agents.
*   **Planning Tool:** An internal tool for formulating and tracking an agent's action plan.
*   **File System Tools:** Built-in tools (`ls`, `edit_file`, `read_file`, `write_file`) that emulate a file system using LangGraph's State object.
*   **Sub Agents:** Facilitates the calling of sub-agents for context isolation, and improved instruction execution.

## Human-in-the-Loop Tool Interrupts

Configure human approval for tool executions using the `interrupt_config` parameter:

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

## Integrating with MCP Tools

Integrate with [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters) to integrate with MCP tools.

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
*   \[ ] Benchmark the example of \[deep research agent](examples/research/research_agent.py)