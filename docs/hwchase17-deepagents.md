# 🧠 Deep Agents: Build Advanced AI Agents for Complex Tasks

**Unleash the power of AI to solve intricate problems with Deep Agents, a Python package designed to create sophisticated agents capable of planning, utilizing tools, and managing sub-agents, inspired by architectures like Claude Code.** Learn more and contribute on the original repo: [https://github.com/hwchase17/deepagents](https://github.com/hwchase17/deepagents).

**Key Features:**

*   🛠️ **Tool Integration:** Easily integrate and utilize a wide range of tools to empower your agents.
*   📝 **Customizable Instructions:** Tailor the agent's behavior and focus with flexible instructions.
*   🤝 **Sub-Agent Support:** Delegate tasks and manage complexity with built-in support for sub-agents.
*   💾 **Virtual File System:** Use a virtual file system with built-in tools to handle file operations securely.
*   💬 **Human-in-the-Loop Approval:** Implement tool execution safeguards with human-in-the-loop approval and customization options.
*   🌐 **LangGraph Integration:** Built on LangGraph, seamlessly integrate with LangGraph's advanced features.
*   🧪 **MCP Support:** Seamlessly integrate and execute MCP tools.
*   🔄 **Customizable Models:** Supports custom LLM model selection, including per-subagent overrides.

## Installation

```bash
pip install deepagents
```

## Usage

**Example: Simple Research Agent**

(Requires `pip install tavily-python`)

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

For a more complex example, see [examples/research/research_agent.py](examples/research/research_agent.py).  The agent created with `create_deep_agent` is a LangGraph graph, enabling full interaction with LangGraph features (streaming, human-in-the-loop, memory, studio).

## Creating a Custom Deep Agent

The `create_deep_agent` function takes three primary parameters, providing extensive customization options:

### `tools` (Required)

*   A list of functions or LangChain `@tool` objects that the agent (and its subagents) can use.

### `instructions` (Required)

*   A string that becomes part of the agent's prompt, guiding its behavior. A built-in system prompt is also included.

### `subagents` (Optional)

*   Allows you to define custom sub-agents with their own instructions, tools, and model settings.
    *   **name**: The name used to call the subagent.
    *   **description**: Description of the subagent for the main agent.
    *   **prompt**: The instructions for the subagent.
    *   **tools**:  Tools available to the subagent (defaults to all tools).
    *   **model_settings**:  Optional model configuration for the subagent.

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

Customize the LLM used by your agent.  Defaults to `claude-sonnet-4-20250514`.

#### Example: Using a Custom Model

```python
from deepagents import create_deep_agent
from langchain_ollama import init_chat_model

model = init_chat_model(
    model="ollama:gpt-oss:20b",  
)
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
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

## Deep Agent Components

`deepagents` incorporates several key components to enhance agent capabilities:

### System Prompt

A comprehensive, built-in system prompt (found in `src/deepagents/prompts.py`) derived from Claude Code's system prompt. This prompt provides detailed instructions for tool usage, planning, file system interaction, and sub-agent utilization.  The `instructions` parameter allows you to customize part of this.

### Planning Tool

A built-in planning tool, inspired by Claude Code's TodoWrite tool. This tool aids the agent in formulating and tracking its plan.

### File System Tools

Four built-in file system tools: `ls`, `edit_file`, `read_file`, and `write_file`.  These tools mock a file system using LangGraph's State object, enabling safe and isolated file operations.

```python
agent = create_deep_agent(...)

result = agent.invoke({
    "messages": ...,
    # Pass in files to the agent using this key
    # "files": {"foo.txt": "foo", ...}
})

# Access any files afterwards like this
result["files"]
```

### Sub Agents

Built-in support for sub-agents, including a `general-purpose` sub-agent with the same instructions and tools as the main agent. You can also define custom sub-agents with unique instructions and tool sets.  Sub agents are used for "context quarantine" and custom instructions.

### Tool Interrupts

Implement human-in-the-loop approval for tool execution using the `interrupt_config` parameter, with these options:
    *   `allow_ignore`: Skip the tool call
    *   `allow_respond`: Add a text response
    *   `allow_edit`: Edit the tool arguments
    *   `allow_accept`: Accept the tool call

```python
from deepagents import create_deep_agent
from langgraph.prebuilt.interrupt import HumanInterruptConfig

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

(Requires `pip install langchain-mcp-adapters`)

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

*   [ ] Allow users to customize the full system prompt
*   [ ] Code cleanliness (type hinting, docstrings, formatting)
*   [ ] Implement a more robust virtual filesystem
*   [ ] Create an example of a deep coding agent
*   [ ] Benchmark the deep research agent example.