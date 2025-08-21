# 🧠 Deep Agents: Build Powerful, Multi-Step AI Agents with Ease

**Unleash the power of deep agents capable of complex reasoning and task execution with `deepagents`, a Python library inspired by cutting-edge research.** ([Original Repo](https://github.com/hwchase17/deepagents))

Deep agents are revolutionizing AI by enabling LLMs to plan, utilize tools, and execute intricate tasks that surpass the limitations of simple agents. `deepagents` provides a streamlined, general-purpose framework to build such agents, incorporating key elements like planning, sub-agents, a virtual file system, and detailed prompts.

**Key Features:**

*   🛠️ **Modular Tool Integration:** Seamlessly integrate custom tools or leverage built-in tools for web search and file system interactions.
*   🧠 **Advanced Planning:** Includes a built-in planning tool to guide agents through complex tasks, inspired by Claude Code's "TodoWrite".
*   👥 **Sub-Agent Support:** Create specialized sub-agents with tailored instructions and tool access for context management and task decomposition.
*   📝 **Flexible Prompting:** Customize agent behavior with instruction parameters and leverage a powerful, adaptable built-in system prompt.
*   💾 **Virtual File System:** Utilize a mock file system with tools like `ls`, `edit_file`, `read_file`, and `write_file` to manage agent-created content within the LangGraph State object.
*   🔄 **LangGraph Compatibility:** Built on top of LangGraph for flexible integration with features like streaming, human-in-the-loop, memory, and LangChain studio.
*   🌐 **MCP Integration:** Use `deepagents` with MCP tools via the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).
*   🚀 **Customizable Models:** Supports customization with various models.
*   ⚙️ **Per-Subagent Model Overrides:** Allows to use different models in the sub agents.

## Installation

```bash
pip install deepagents
```

## Quickstart

```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

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

research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

agent = create_deep_agent(
    [internet_search],
    research_instructions,
)

result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
```

Explore a more advanced example in [examples/research/research_agent.py](examples/research/research_agent.py).

## Customizing Your Deep Agent

`create_deep_agent` provides the following parameters for customization:

### `tools` (Required)

A list of functions or LangChain `@tool` objects that the agent and sub-agents can utilize.

### `instructions` (Required)

Custom instructions to guide the agent's behavior, combined with a [built-in system prompt](src/deepagents/prompts.py).

### `subagents` (Optional)

Define custom sub-agents with specific instructions and tool access.

```python
from typing import TypedDict, NotRequired, Any

class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    model_settings: NotRequired[dict[str, Any]]
```

*   `name`: The name of the subagent.
*   `description`: A description of the subagent.
*   `prompt`: The prompt for the subagent.
*   `tools`: The tools the subagent has access to (defaults to all).
*   `model_settings`: Per-subagent model configuration.

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

Defaults to `"claude-sonnet-4-20250514"`.  You can use other models.

#### Example: Custom Model

```python
from deepagents import create_deep_agent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage

class CustomRunnable(Runnable):
    ...
```

```python
model = CustomRunnable()

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

## Deep Agent Components

`deepagents` integrates these components to enable "deep" agent capabilities:

### System Prompt

A built-in system prompt, heavily inspired by Claude Code, provides detailed instructions for utilizing planning, file system tools, and sub-agents.  It is designed to enhance agent performance.

### Planning Tool

A built-in planning tool, inspired by Claude Code's TodoWrite, to guide agents through task decomposition and execution.

### File System Tools

Built-in file system tools (`ls`, `edit_file`, `read_file`, `write_file`) that mock a virtual file system using LangGraph's State object, allowing agents to manage files within the context of the task.

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

The ability to call sub-agents, including a default `general-purpose` subagent, as well as custom sub-agents, to facilitate context quarantine and specialized instruction sets.

## MCP Integration

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