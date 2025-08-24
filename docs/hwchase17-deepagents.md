# 🧠🤖 Deep Agents: Build Advanced AI Agents for Complex Tasks

**Unlock the power of advanced AI agents capable of planning, researching, and executing intricate tasks with `deepagents`.**

`deepagents` is a Python package that allows you to easily create powerful AI agents that can tackle complex, multi-step projects.  Inspired by the architecture of advanced agents like Claude Code,  `deepagents` provides the building blocks to create agents that go beyond simple tool calls, enabling them to plan, utilize file systems, and leverage sub-agents for nuanced problem-solving.  Check out the original repo for more inspiration and context: [https://github.com/hwchase17/deepagents](https://github.com/hwchase17/deepagents)

**Key Features:**

*   **Planning Tool:**  Built-in planning capabilities to guide your agent's actions.
*   **Sub-Agents:** Modularize complex tasks and enable specialized problem-solving.
*   **File System Tools:** Mock file system for reading, writing, and editing files.
*   **Customizable Prompts:** Tailor instructions to guide your agent's behavior.
*   **Human-in-the-Loop Support:**  Integrate human approval for tool execution with `interrupt_config`.
*   **MCP Integration:** Run `deepagents` with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

## Installation

```bash
pip install deepagents
```

## Usage

(Requires `pip install tavily-python` for the following example)

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

For a more advanced example, see [examples/research/research_agent.py](examples/research/research_agent.py).

`create_deep_agent` produces a LangGraph graph, offering seamless integration with LangGraph features like streaming, human-in-the-loop interaction, memory, and Studio.

## Customizing Your Deep Agent

Customize your agents using the `create_deep_agent` function with these parameters:

### `tools` (Required)

Provide a list of functions or LangChain `@tool` objects.  These tools empower your agent.

### `instructions` (Required)

Supply the primary instructions for your agent. `deepagents` combines these with a built-in system prompt (see [Deep Agent Details](#deep-agent-details) for more details).

### `subagents` (Optional)

Define custom sub-agents for specialized tasks.  Sub-agents can provide context quarantine and custom instructions. The structure for `subagents` is as follows:

```python
from typing import Any, NotRequired, TypedDict

class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    model_settings: NotRequired[dict[str, Any]]
```

Example:

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

By default, `deepagents` uses `"claude-sonnet-4-20250514"`. Use any [LangChain model object](https://python.langchain.com/docs/integrations/chat/) to customize this.  Install  `pip install langchain` and then `pip install langchain-ollama` for Ollama models.

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

## Deep Agent Details

These components are built into `deepagents` to create powerful agents:

### System Prompt

`deepagents` includes a detailed [built-in system prompt](src/deepagents/prompts.py) that guides agent behavior, inspired by Claude Code and tailored for general use.  The prompts provide detailed instructions for using planning, file system tools, and sub-agents.  Part of this prompt is customizable via the `instructions` parameter.

### Planning Tool

The integrated planning tool, based on ClaudeCode's TodoWrite tool, enables agents to create and maintain plans.

### File System Tools

`deepagents` provides four built-in file system tools: `ls`, `edit_file`, `read_file`, and `write_file`.  These tools use LangGraph's State object to simulate a file system, allowing you to easily manage files.

Files can be passed in and retrieved via the `files` key in the LangGraph State object.

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

`deepagents` allows agents to utilize sub-agents, including a default `general-purpose` agent with the same tools and instructions as the main agent. You can also define [custom sub agents](#subagents-optional) with distinct instructions and toolsets.

### Tool Interrupts

Implement human-in-the-loop approval for tool execution. Configure tools with the `interrupt_config` parameter:

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

The `interrupt_config` uses four parameters: `allow_ignore`, `allow_respond`, `allow_edit`, and `allow_accept`.  When approval is needed, the agent pauses, waits for human input, and displays a message with the tool's details.

## Using deepagents with MCP tools

The `deepagents` library can be ran with MCP tools. This can be achieved by using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

(To run the example below, will need to `pip install langchain-mcp-adapters`)

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
*   \[ ] Develop a more robust virtual filesystem.
*   \[ ] Create a deep coding agent example.
*   \[ ] Benchmark the example of a deep research agent ([examples/research/research_agent.py](examples/research/research_agent.py)).