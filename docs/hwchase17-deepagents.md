# 🧠🤖 Deep Agents: Build Powerful, Recursive AI Agents with Ease

**Create intelligent agents that can plan, research, and execute complex tasks like never before with `deepagents`!**  This Python package offers a streamlined approach to building advanced AI agents, inspired by cutting-edge architectures like Claude Code, by combining planning, sub-agents, file system access, and detailed prompts.

[![GitHub stars](https://img.shields.io/github/stars/hwchase17/deepagents?style=social)](https://github.com/hwchase17/deepagents)
[![PyPI version](https://badge.fury.io/py/deepagents.svg)](https://badge.fury.io/py/deepagents)

<img src="deep_agents.png" alt="deep agent" width="600"/>

**Key Features:**

*   **Simplified Agent Creation:** Easily build deep agents with minimal setup using `create_deep_agent()`.
*   **Planning and Execution:** Leverage a built-in planning tool (inspired by Claude Code's TodoWrite) to guide agent actions.
*   **Virtual Filesystem:** Includes built-in tools (`ls`, `edit_file`, `read_file`, `write_file`) that mock a filesystem using LangGraph's State object, for data management during agent execution.
*   **Sub-Agent Support:** Implement modular sub-agents with custom instructions and tools to tackle complex tasks, promoting context isolation.
*   **Human-in-the-Loop Integration:** Integrate human oversight and approval with the `interrupt_config` parameter.
*   **Customizable Prompts:** Tailor the agent's behavior by providing custom instructions and leveraging the built-in system prompt (inspired by Claude Code).
*   **Flexible Tool Integration:** Integrate custom tools (functions or LangChain `@tool` objects) with ease.
*   **Model Customization:** Utilize different LLM models from LangChain via the `model` parameter and configure per-subagent model overrides.
*   **Asynchronous Support:** Easily use `async_create_deep_agent` to support your asynchronous tools.
*   **MCP Compatibility:** Runs with MCP tools via Langchain MCP Adapter library.
*   **Configurable Agent:** Built-in support for configurable agents by passing a config via json.

## Installation

```bash
pip install deepagents
```

## Quick Start

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

For a more complex example, see [examples/research/research_agent.py](examples/research/research_agent.py).

## Core Concepts and Customization

`deepagents` uses LangGraph under the hood, giving you access to all of LangGraph's features.

### Creating a Custom Deep Agent

You can tailor your deep agent using these parameters:

*   **`tools` (Required):** A list of functions or LangChain `@tool` objects.
*   **`instructions` (Required):**  Custom prompt to steer the agent (combines with a built-in system prompt).
*   **`subagents` (Optional):** Create custom sub-agents with specific instructions and tools.
*   **`model` (Optional):** Customize the underlying LLM model used by the agent.
*   **`builtin_tools` (Optional):** Control which built-in tools the agent has access to.

### Deep Agent Details

`deepagents` incorporates the following:

*   **System Prompt:**  A detailed built-in system prompt (inspired by Claude Code) to manage planning, file system interactions, and sub-agent calls.  The system prompt is found in [src/deepagents/prompts.py](src/deepagents/prompts.py).
*   **Planning Tool:** A built-in tool for the agent to create a plan.
*   **File System Tools:**  Virtual file system tools (`ls`, `edit_file`, `read_file`, `write_file`) built on LangGraph State.
*   **Sub Agents:** A `general-purpose` subagent available and configurable [custom sub agents](#subagents-optional) for modularity.
*   **Built-In Tools:**  Provides tools for writing todos, managing a virtual file system, and editing files.  Disabled via the `builtin_tools` parameter.

### Human-in-the-Loop

Integrate human approval for tool execution:

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

# Create agent with file operations requiring approval
agent = create_deep_agent(
    tools=[your_tools],
    instructions="Your instructions here",
    interrupt_config={
        # You can specify a dictionary for fine grained control over what interrupt options exist
        "tool_1": {
            "allow_ignore": False,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept":True,
        },
        # You can specify a boolean for shortcut
        # This is a shortcut for the same functionality as above
        "tool_2": True,
    }
)

checkpointer= InMemorySaver()
agent.checkpointer = checkpointer
```

### Async

For asynchronous tools, use `from deepagents import async_create_deep_agent`.

## Examples

### Custom Model
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
### Per-subagent model override (optional)
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

### MCP
```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

async def main():
    # Collect MCP tools
    mcp_client = MultiServerMCPClient(...)
    mcp_tools = await mcp_client.get_tools()

    # Create agent
    agent = async_create_deep_agent(tools=mcp_tools, ....)

    # Stream the agent
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is langgraph?"}]},
        stream_mode="values"
    ):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()

asyncio.run(main())
```

### Configurable Agent

```python
from deepagents import create_configurable_agent

agent_config = {"instructions": "foo", "subagents": []}

build_agent = create_configurable_agent(
    agent_config['instructions'],
    agent_config['subagents'],
    [],
    agent_config={"recursion_limit": 1000}
)
```
## Contribute

The `deepagents` library is open-source under the MIT license, available on [GitHub](https://github.com/hwchase17/deepagents). Contributions are welcome!

## Roadmap
- [ ] Allow users to customize full system prompt
- [ ] Code cleanliness (type hinting, docstrings, formating)
- [ ] Allow for more of a robust virtual filesystem
- [ ] Create an example of a deep coding agent built on top of this
- [ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)