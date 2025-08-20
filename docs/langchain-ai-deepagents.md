# 🧠🤖 Deep Agents: Build Intelligent AI Agents with Planning and Sub-Agents

**Unlock the power of deep agents to tackle complex tasks by enabling planning, sub-agents, and file system access.**  

[Go to the Deep Agents Repository](https://github.com/langchain-ai/deepagents)

Deep Agents go beyond simple LLM-based agents by incorporating advanced capabilities found in applications like "Deep Research" and "Claude Code." The `deepagents` Python package provides a streamlined way to create these sophisticated agents, incorporating the following key features:

*   **Planning Tool:** Enables the agent to strategize and create a plan.
*   **Sub-Agents:** Allows agents to delegate tasks and manage context effectively.
*   **File System Access:** Provides a virtual file system for storing and retrieving information, enhancing the agent's knowledge base.
*   **Customizable Prompts:** Empowers you to tailor the agent's behavior and expertise with detailed instructions.

## Installation

```bash
pip install deepagents
```

## Usage

(To run the example below, will need to `pip install tavily-python`)

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

The agent created with `create_deep_agent` is just a LangGraph graph - so you can interact with it (streaming, human-in-the-loop, memory, studio)
in the same way you would any LangGraph agent.

## Creating a Custom Deep Agent

Customize your deep agent with the `create_deep_agent` function, utilizing the following parameters:

### `tools` (Required)

Pass a list of functions or LangChain `@tool` objects to give the agent (and any subagents) access to these tools.

### `instructions` (Required)

Provide a custom prompt that is combined with a [built-in system prompt](src/deepagents/prompts.py). This tailors the agent's behavior.

### `subagents` (Optional)

Define custom subagents to handle specific tasks, improving context management and task delegation. Use the following schema:

```python
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

Customize the language model used by the agent. By default, `"claude-sonnet-4-20250514"` is used, but you can use any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

#### Example: Using a Custom Model

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

#### Example: Per-subagent model override (optional)

Use a fast, deterministic model for a critique sub-agent, while keeping a different default model for the main agent and others:

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

These components are built into `deepagents` for out-of-the-box functionality.

### System Prompt

`deepagents` incorporates a robust [built-in system prompt](src/deepagents/prompts.py) inspired by Claude Code. This prompt provides detailed instructions for the planning tool, file system tools, and sub-agents, driving the agent's success.

### Planning Tool

`deepagents` includes a built-in planning tool based on Claude Code's TodoWrite tool, facilitating task organization and execution.

### File System Tools

The library provides four built-in file system tools: `ls`, `edit_file`, `read_file`, and `write_file`. They use LangGraph's State object to mock out a file system, allowing you to run multiple agents without file conflicts.

Example:

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

`deepagents` supports the use of sub-agents, inspired by Claude Code, for complex task management and context quarantine. A `general-purpose` subagent is always available, and you can define custom subagents.

## MCP Integration

Integrate `deepagents` with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

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
*   \[ ] Improve code quality through type hinting, docstrings, and formatting.
*   \[ ] Enhance the virtual file system.
*   \[ ] Create a deep coding agent example.
*   \[ ] Benchmark the deep research agent example.
*   \[ ] Add human-in-the-loop support for tools.