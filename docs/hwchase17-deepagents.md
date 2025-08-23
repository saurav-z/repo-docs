# 🧠🤖 DeepAgents: Build Powerful, Deep Agents with Ease

**Unleash the power of advanced AI agents by leveraging planning, sub-agents, and file system access for complex tasks.**  [View the project on GitHub](https://github.com/hwchase17/deepagents)

DeepAgents is a Python package that enables you to create sophisticated, "deep" agents capable of handling intricate, multi-step tasks that go beyond simple prompt-tool interactions. Inspired by cutting-edge applications like Claude Code, DeepAgents provides a flexible framework built on LangGraph to build agents with advanced features.

**Key Features:**

*   **Planning Tool:** Built-in planning mechanism to guide the agent's actions.
*   **Sub-Agents:** Enables modularity and specialization through the use of custom and general purpose sub-agents for handling specific tasks.
*   **File System Tools:**  Includes `ls`, `edit_file`, `read_file`, and `write_file` tools using a mocked, state-based file system.
*   **Detailed Prompting:** Leverage a powerful, customizable system prompt to guide agent behavior, based on Claude Code principles.
*   **Human-in-the-Loop:** Support for tool interrupt/approval using the `interrupt_config` parameter.
*   **MCP Integration:** Compatible with the Langchain MCP adapter for integration with MCP tools.
*   **Customization:**  Easily customize agents with tools, instructions, and sub-agents.
*   **Model Flexibility:**  Supports LangChain model objects, allowing use of diverse LLMs including OpenAI and Ollama models, with per-subagent model overrides.

## Installation

```bash
pip install deepagents
```

## Usage

(Requires `pip install tavily-python` for the example below)

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
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

research_instructions = """You are an expert researcher. Conduct research and write a report.

You have access to a few tools.
## `internet_search`
Use this to run an internet search."""

# Create the agent
agent = create_deep_agent(
    [internet_search],
    research_instructions,
)

# Invoke the agent
result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
```

For a more complex example, see: [examples/research/research_agent.py](examples/research/research_agent.py).

## Customizing Your Deep Agent

The `create_deep_agent` function provides flexibility to tailor your agents to specific needs:

### `tools` (Required)

Pass a list of functions or LangChain `@tool` objects to grant the agent access to specific functionalities.

### `instructions` (Required)

Provide the agent's prompt instructions to guide its overall behavior.  A built-in system prompt is also used.

### `subagents` (Optional)

Define custom sub-agents with specific instructions and tools:

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

Customize the LLM used by the agent.  By default, `"claude-sonnet-4-20250514"` is used.  You can supply any LangChain model object. Per-subagent model overrides are also possible.

#### Example: Using a Custom Model (Ollama)

```python
from deepagents import create_deep_agent
from langchain_community.llms import Ollama

model = Ollama(model="gpt-oss:20b")
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
    ...
)
```

#### Example: Per-Subagent Model Override

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

## Deep Agent Architecture

`deepagents` builds upon LangGraph and provides several built-in components for creating effective agents:

### System Prompt

A detailed system prompt (available in [src/deepagents/prompts.py](src/deepagents/prompts.py)), inspired by Claude Code, provides instructions for planning, using file system tools, and utilizing sub-agents.

### Planning Tool

A basic planning tool, similar to Claude Code's TodoWrite, helps the agent strategize and stay on track.

### File System Tools

A set of file system tools (`ls`, `edit_file`, `read_file`, `write_file`) provide a mocked file system using LangGraph's State object.

### Sub Agents

Supports calling sub-agents, including a built-in general-purpose sub-agent and customizable sub-agents, for context quarantine and specialized instructions.

### Tool Interrupts

Human-in-the-loop approval for tool execution using `interrupt_config`.

## Integration

### MCP

Integrates with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

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

*   \[ ] Allow users to customize the full system prompt.
*   \[ ] Code cleanliness (type hinting, docstrings, formatting).
*   \[ ] Enhance the virtual filesystem.
*   \[ ] Develop a deep coding agent example.
*   \[ ] Benchmark the deep research agent example.