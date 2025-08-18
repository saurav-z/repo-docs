# Deep Agents: Build Powerful, Intelligent Agents with Ease

**Unlock the potential of advanced AI agents with `deepagents`, a Python package designed to create sophisticated agents capable of tackling complex, multi-step tasks.**  ([Original Repo](https://github.com/hwchase17/deepagents))

`deepagents` empowers you to build agents that go beyond simple tool calls, offering features inspired by cutting-edge research like Claude Code.

**Key Features:**

*   **Planning Tool:**  Built-in tool to help your agent strategize and stay on track.
*   **Sub-Agent Support:**  Create modular agents with context quarantine and custom instructions for specific sub-tasks.
*   **Virtual File System:** Mock file system to simulate file operations without modifying the real file system.
*   **Customizable Prompts:** Fine-tune agent behavior with custom instructions.
*   **Model Flexibility:** Use any LangChain model, including OpenAI and Ollama.
*   **LangGraph Integration:** Deep agents created with `create_deep_agent` are just a LangGraph graph - interact with them the same way you would any LangGraph agent.
*   **MCP Compatibility:** Integrate your agents with MCP tools using the Langchain MCP Adapter library.

## Installation

```bash
pip install deepagents
```

## Usage

Here's a quick example demonstrating a research agent using the Tavily API:

**(Requires `pip install tavily-python` and a Tavily API key)**

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

Explore a more complex example in [examples/research/research_agent.py](examples/research/research_agent.py).

## Creating a Custom Deep Agent

Customize your deep agent by passing the following parameters to `create_deep_agent`:

### `tools` (Required)

A list of functions or LangChain `@tool` objects accessible to the agent and subagents.

### `instructions` (Required)

Instructions to shape the agent's behavior. This is combined with a built-in system prompt.

### `subagents` (Optional)

Define custom sub-agents with their own instructions and tools for context quarantine.

```python
from deepagents import create_deep_agent

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

Specify any [LangChain model object](https://python.langchain.com/docs/integrations/chat/) to customize your agent's language model. Default is `"claude-sonnet-4-20250514"`.

#### Example: Using a Custom Model (Ollama)

```python
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
# ... existing agent definitions ...

model = ChatOllama(model="gpt-oss:20b")
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
    ...
)
```

## Deep Agent Architecture

`deepagents` incorporates key components to facilitate deep, multi-step tasks:

### System Prompt

A detailed, built-in system prompt inspired by Claude Code's system prompt. Contains detailed instructions for using the built-in planning tool, file system tools, and sub agents. The system prompt improves the agents ability to go deep.

### Planning Tool

A built-in planning tool to help the agent come up with a plan.

### File System Tools

Built-in tools (`ls`, `edit_file`, `read_file`, `write_file`) simulate a file system using LangGraph's State object.  Files are passed and retrieved using the `"files"` key in the LangGraph State object.

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

Supports sub-agents with their own prompts and tools for context isolation and specialized tasks.  A `general-purpose` subagent is available with the same instructions and tools as the main agent.

## MCP Integration

Integrate `deepagents` with [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters) and MCP tools.

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

*   \[ ] Allow users to customize the full system prompt
*   \[ ] Code Cleanliness (type hinting, docstrings, formating)
*   \[ ] Allow for a more robust virtual filesystem
*   \[ ] Create an example of a deep coding agent built on top of this
*   \[ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)
*   \[ ] Add human-in-the-loop support for tools