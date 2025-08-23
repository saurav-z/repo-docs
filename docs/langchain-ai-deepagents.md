# DeepAgents: Build Powerful, Deep Agents with LLMs

**Unlock the potential of complex tasks with DeepAgents, a Python package designed to create sophisticated agents capable of planning, utilizing tools, and achieving multi-step goals.**  [See the original repo](https://github.com/langchain-ai/deepagents)

DeepAgents empower LLMs to call tools in a loop for complex tasks. Key applications include "Deep Research," "Manus," and "Claude Code."

**Key Features:**

*   **Planning Tool:** Built-in planning tool based on Claude Code's TodoWrite.
*   **Sub-Agent Support:** Enables context quarantine and custom instructions for advanced task decomposition.
*   **File System Tools:** Mock file system using LangGraph's State object for `ls`, `edit_file`, `read_file`, and `write_file` operations.
*   **Customizable System Prompt:** Includes a default system prompt with detailed instructions and the ability to customize.
*   **Tool Interrupts:**  Supports human-in-the-loop approval for tool execution with configurable options for allowing the user to ignore, respond, edit or accept the tool call.
*   **MCP Support:** Compatible with MCP tools via the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

## Installation

```bash
pip install deepagents
```

## Usage Examples

### Basic Example
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

### Custom Deep Agents

`create_deep_agent` allows customization through three key parameters:

*   **`tools` (Required):** A list of functions or LangChain `@tool` objects.
*   **`instructions` (Required):**  Custom instructions to guide the agent's behavior.
*   **`subagents` (Optional):** Define custom subagents with their own instructions and tools (see below).
*   **`model` (Optional):** Customize the LLM used by the agent (defaults to `"claude-sonnet-4-20250514"`).

### Using Sub-agents

Sub-agents enable context quarantine and specialized instructions for tasks.

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

### Tool Interrupts (Human-in-the-Loop)

Configure specific tools to require human approval before execution:
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

## Deep Agent Details

*   **System Prompt:** Built-in prompt based on Claude Code for effective task execution.  (Found in `src/deepagents/prompts.py`)
*   **Planning Tool:** A simplified planning tool based on Claude Code.
*   **File System Tools:**  Mock file system using LangGraph State for `ls`, `edit_file`, `read_file`, and `write_file`.
*   **Sub Agents:** Built-in ability to call sub agents for complex task decomposition and custom instructions.
*   **Tool Interrupts**: Supports human approval before tool execution with customizable configurations.

## Integration with MCP

DeepAgents can integrate with MCP tools using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

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

*   Allow users to customize the full system prompt.
*   Improve code cleanliness (type hinting, docstrings, formatting).
*   Implement a more robust virtual file system.
*   Develop a deep coding agent example.
*   Benchmark the deep research agent example.