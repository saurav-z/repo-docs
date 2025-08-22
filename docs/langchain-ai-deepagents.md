# Deep Agents: Build Intelligent Agents with Enhanced Planning and Complex Task Handling

**Unlock advanced agent capabilities with `deepagents`, a Python package enabling sophisticated LLM-powered agents capable of tackling complex tasks.**  [View the GitHub Repository](https://github.com/langchain-ai/deepagents)

## Key Features:

*   **Deep Planning:** Enables agents to strategize and execute tasks over longer durations.
*   **Sub-Agents:** Facilitates modular design and context management through the use of specialized sub-agents.
*   **Built-in File System:** Provides a virtual file system for persistent context and data management.
*   **Customizable Prompts:** Allows tailoring instructions to guide agent behavior effectively.
*   **Tool Interrupts:** Supports human-in-the-loop approval for tool execution.
*   **LangGraph Integration:** Leverages LangGraph for seamless integration with LangChain features.
*   **MCP Tool Compatibility:** Works with MCP tools via the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters)

## Installation

```bash
pip install deepagents
```

## Getting Started

`deepagents` allows you to create advanced agents with a few lines of code.

(To run the example below, you will need `pip install tavily-python`)

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


# Prompt prefix to guide the agent to be an expert researcher
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

For a more advanced use case, explore the [examples/research/research_agent.py](examples/research/research_agent.py) file.

## Customizing Your Deep Agent

The `create_deep_agent` function offers several parameters for tailoring your agent:

### `tools` (Required)

*   Provide a list of functions or LangChain `@tool` objects that the agent will utilize.

### `instructions` (Required)

*   Define the core instructions that will shape the agent's behavior. See the [built-in system prompt](src/deepagents/prompts.py) for default instructions.

### `subagents` (Optional)

*   Use this parameter to define and incorporate custom sub-agents.  Sub-agents are defined with a schema including `name`, `description`, `prompt`, and optional `tools` and `model_settings`.

    ```python
    class SubAgent(TypedDict):
        name: str
        description: str
        prompt: str
        tools: NotRequired[list[str]]
        model_settings: NotRequired[dict[str, Any]]
    ```

### `model` (Optional)

*   Customize the LLM used by the agent. By default `"claude-sonnet-4-20250514"` is used. You can pass any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

    ```python
    from deepagents import create_deep_agent

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

    *   **Per-subagent Model Override (Optional):** You can override the model for specific sub-agents.

        ```python
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

`deepagents` incorporates several key components for enhanced performance:

### System Prompt

*   Leverages a built-in system prompt inspired by Claude Code, providing detailed instructions for planning, file system interactions, and sub-agent use.

### Planning Tool

*   Integrates a built-in planning tool (similar to Claude Code's TodoWrite tool) to facilitate task decomposition.

### File System Tools

*   Provides built-in file system tools (`ls`, `edit_file`, `read_file`, `write_file`) which are mocked using LangGraph's State object.

### Sub Agents

*   Offers the ability to invoke sub-agents.  Includes a `general-purpose` subagent by default.
*   Supports [custom sub-agents](#subagents-optional) for context isolation and specialized instructions.

### Tool Interrupts

*   Support human-in-the-loop approval for tool execution. Configure specific tools using the `interrupt_config` parameter.

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

## Integration with MCP Tools

The `deepagents` library is compatible with MCP tools by using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

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

*   [ ] Allow users to customize full system prompt
*   [ ] Code cleanliness (type hinting, docstrings, formating)
*   [ ] Allow for more of a robust virtual filesystem
*   [ ] Create an example of a deep coding agent built on top of this
*   [ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)