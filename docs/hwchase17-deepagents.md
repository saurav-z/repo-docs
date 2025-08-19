# Deep Agents: Build Powerful, Multi-Step AI Agents with Python

**Unleash the power of advanced AI agents with `deepagents`, a Python package designed to create sophisticated agents capable of complex tasks through planning, sub-agents, and tool utilization.** Explore the capabilities of [Deep Agents](https://github.com/hwchase17/deepagents) and learn how to implement them for your use case.

## Key Features

*   **Planning & Execution:** Utilize a built-in planning tool to strategize and manage complex tasks effectively.
*   **Sub-Agent Support:** Design and integrate sub-agents for context isolation and specialized functionalities.
*   **Virtual File System:** Emulate a file system with built-in tools to interact with and manage virtual files.
*   **Customizable & Extensible:** Tailor agents with custom instructions, tools, and sub-agents.
*   **LangGraph Integration:** Leverage the power of LangGraph for easy interaction (streaming, human-in-the-loop, memory, studio)
*   **MCP Compatibility:** Integrate with LangChain MCP tools for extended functionality.

## Installation

Install `deepagents` using pip:

```bash
pip install deepagents
```

## Getting Started

Here's how to create a deep agent:

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

## Customization Options

### Tools (Required)

Pass a list of functions or LangChain `@tool` objects to define the tools your agent will use.

### Instructions (Required)

Provide a string of instructions to steer your agent's behavior; this is combined with a [built-in system prompt](src/deepagents/prompts.py) for enhanced performance.

### Sub-Agents (Optional)

Define custom sub-agents with their own instructions, tools, and context to improve task-specific performance.

```python
# Example of Custom Sub-Agents
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

### Model (Optional)

By default, `deepagents` uses `"claude-sonnet-4-20250514"`. You can customize this by passing any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

#### Example: Using a Custom Model

Here's how to use a custom model (like OpenAI's `gpt-oss` model via Ollama):

(Requires `pip install langchain` and then `pip install langchain-ollama` for Ollama models)

```python
from deepagents import create_deep_agent

# ... existing agent definitions ...

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

## Deep Agent Core Components

*   **System Prompt:** A comprehensive built-in system prompt, inspired by Claude Code, guides the agent's behavior.
*   **Planning Tool:** A built-in tool helps the agent create and follow a plan.
*   **File System Tools:** Built-in `ls`, `edit_file`, `read_file`, and `write_file` tools for interacting with a virtual file system using the LangGraph State object.
*   **Sub-Agents:** Access to a general-purpose subagent and the ability to customize subagents.

## MCP Integration

Integrate `deepagents` with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

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