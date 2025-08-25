# DeepAgents: Build Advanced AI Agents with Planning, Subagents, and Tools

**Tackle complex tasks with ease! DeepAgents empowers you to create powerful AI agents capable of planning, utilizing subagents, and interacting with tools like never before.** ([Original Repository](https://github.com/hwchase17/deepagents))

DeepAgents builds upon the concept of agents that use tools in a loop but enhances them to handle intricate, long-term tasks.  It achieves this by incorporating: planning tools, subagents, file system interaction, and a detailed, adaptable prompt system.

**Key Features:**

*   **Modular Agent Creation:** Easily create deep agents by combining tools, instructions, and subagents.
*   **Built-in System Prompt:**  Leverages a comprehensive system prompt designed to guide agents through complex tasks. (Inspired by Claude Code)
*   **Planning Tool:** Includes a basic planning tool to enable agents to strategize and maintain focus.
*   **Virtual File System:** Offers built-in tools (`ls`, `edit_file`, `read_file`, `write_file`) for file interaction within a sandboxed environment.
*   **Subagent Support:**  Facilitates the use of subagents for context isolation ("context quarantine") and specialized tasks, enhancing the agent's capabilities.
*   **Tool Interrupts:** Enables human-in-the-loop approval for critical tool calls.
*   **MCP Integration:** Supports Multi-Server MCP client via the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).
*   **Customizable Models:** Supports model customization via LangChain models, including per-subagent model overrides for fine-grained control (e.g., faster models for specific subagents).

## Installation

```bash
pip install deepagents
```

## Usage

Below is a simple example of how to create and invoke a deep agent.  You'll need to install the Tavily Search tool for this specific example: `pip install tavily-python`

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
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# Instructions
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# Create agent
agent = create_deep_agent(
    [internet_search],
    research_instructions,
)

# Invoke agent
result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
```

For a more comprehensive example, see `examples/research/research_agent.py`.

The agents created with `create_deep_agent` is a LangGraph graph - so you can interact with it (streaming, human-in-the-loop, memory, studio)
in the same way you would any LangGraph agent.

## Creating a Custom Deep Agent

The `create_deep_agent` function takes the following parameters:

### `tools` (Required)

A list of functions or LangChain `@tool` objects that the agent and its subagents can use.

### `instructions` (Required)

A string providing instructions for the main agent. This is combined with a built-in system prompt.

### `subagents` (Optional)

A list of dictionaries, each defining a custom subagent:

```python
class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    model_settings: NotRequired[dict[str, Any]]
```

*   `name`: Subagent's identifier.
*   `description`: Brief description of the subagent.
*   `prompt`: Instructions for the subagent.
*   `tools`: (Optional) Specific tools for the subagent (defaults to all tools).
*   `model_settings`: (Optional) Per-subagent model configuration.

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

Customize the LLM used by your agents. By default, it uses `"claude-sonnet-4-20250514"`. Specify any [LangChain model object](https://python.langchain.com/docs/integrations/chat/) as the `model` parameter.

Example:

```python
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama

# ... existing agent definitions ...

model = ChatOllama(model="gpt-oss:20b") # or any other model
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
    ...
)
```

## Deep Agent Details

### System Prompt

DeepAgents includes a comprehensive system prompt, inspired by Claude Code. This provides detailed guidance for using the planning tool, file system tools, and subagents, greatly enhancing agent performance.

### Planning Tool

A built-in planning tool (inspired by Claude Code) is included to facilitate task planning.

### File System Tools

Four built-in tools (`ls`, `edit_file`, `read_file`, `write_file`) allow interaction with a virtual file system within the LangGraph state.

### Sub Agents

Supports the creation of subagents with unique instructions and access to tools, and also has a built in `general-purpose` subagent.

### Tool Interrupts

Enables human-in-the-loop approval for tool execution for specific tools, configurable via the `interrupt_config` parameter.

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

## MCP Support

DeepAgents integrates with the LangChain MCP Adapter library for tools using Multi-Server MCP client.

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
- [ ] Allow users to customize full system prompt
- [ ] Code cleanliness (type hinting, docstrings, formating)
- [ ] Allow for more of a robust virtual filesystem
- [ ] Create an example of a deep coding agent built on top of this
- [ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)