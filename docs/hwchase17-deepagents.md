# 🧠🤖 Deep Agents: Build Powerful, Self-Improving Agents with LLMs

**Unlock the potential of advanced AI with `deepagents`, a Python package that allows you to create sophisticated LLM-powered agents capable of complex planning and execution.**  [View the original repo](https://github.com/hwchase17/deepagents)

`deepagents` provides a framework inspired by cutting-edge applications like Claude Code, offering a general-purpose way to build "deep" agents that go beyond simple tool calls. These agents leverage a combination of planning, sub-agents, file system access, and detailed prompts to tackle intricate tasks.

**Key Features:**

*   **Modular Architecture:** Build complex agents by combining planning, sub-agents, and file system tools.
*   **Built-in Planning:** Comes with a default planning tool inspired by Claude Code's TodoWrite to guide the agent's actions.
*   **File System Tools:** Includes `ls`, `edit_file`, `read_file`, and `write_file` for interacting with a virtual file system.
*   **Sub-Agent Support:** Enables context quarantine and specialized task handling with customizable sub-agents.
*   **Customization:** Easily define tools, instructions, and sub-agents to tailor agents to your specific needs.
*   **Human-in-the-Loop (Tool Interrupts):** Integrated support for human approval of tool execution.
*   **MCP Tools Compatibility:** Built-in support for [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

## Installation

```bash
pip install deepagents
```

## Usage

**(Requires `pip install tavily-python` for the example below)**

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

For a more comprehensive example, see [examples/research/research_agent.py](examples/research/research_agent.py).

## Customizing Deep Agents

The `create_deep_agent` function provides flexible options to tailor your agents:

### `tools` (Required)

Pass a list of functions or LangChain `@tool` objects for the agent and sub-agents to use.

### `instructions` (Required)

Define instructions to guide the agent's behavior. These instructions are incorporated into the agent's prompt. Note that there is a [built in system prompt](src/deepagents/prompts.py) as well, so this is not the *entire* prompt the agent will see.

### `subagents` (Optional)

Define sub-agents with their own instructions and tools. This helps in context isolation and specialization.

```python
class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    model_settings: NotRequired[dict[str, Any]]
```

```python
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

By default, `deepagents` uses `"claude-sonnet-4-20250514"`. You can pass any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

#### Example: Using a Custom Model

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

#### Example: Per-subagent model override (optional)

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

## Built-in Deep Agent Components

`deepagents` includes several components to facilitate the creation of robust agents:

### System Prompt

A [built-in system prompt](src/deepagents/prompts.py) inspired by Claude Code, providing detailed instructions for planning, file system tools, and sub-agent usage.

### Planning Tool

A simple planning tool based on Claude Code's TodoWrite, aiding in task planning.

### File System Tools

Mock file system tools (`ls`, `edit_file`, `read_file`, `write_file`) that utilize LangGraph's State object for file management.

### Sub Agents

Built-in support for sub-agents, including a `general-purpose` agent, and the option to define custom sub-agents for task specialization.

### Tool Interrupts

Integrate human approval for tool execution, enhancing control and oversight.

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

## MCP Integration

Integrate `deepagents` with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

**(Requires `pip install langchain-mcp-adapters`)**

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