# 🧠 Deep Agents: Unleash Advanced AI Capabilities

**Build sophisticated, multi-step agents that plan, research, and execute complex tasks with ease.** Dive into the world of "deep agents" that go beyond simple tool calls, leveraging advanced architectures for superior performance.

[Original Repository](https://github.com/hwchase17/deepagents)

**Key Features:**

*   ✅ **Planning Tool:** Integrated planning tool to guide the agent's decision-making process.
*   ✅ **Sub-Agents:** Delegate tasks to specialized sub-agents for focused execution.
*   ✅ **Virtual File System:** Built-in file system tools for storage and retrieval of information.
*   ✅ **Detailed Prompting:** Utilize comprehensive system prompts designed for complex task handling.
*   ✅ **Human-in-the-Loop:** Support for human approval before tool execution, enhancing control and reliability.
*   ✅ **MCP Integration:** Compatible with Multi Server MCP tools for expanded functionality.

## Installation

```bash
pip install deepagents
```

## Getting Started

`deepagents` empowers you to create agents capable of tackling intricate challenges. Below is a basic example, showcasing how to build a research agent that uses a search tool:

**(Requires `pip install tavily-python` for the example)**

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
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# Instructions
research_instructions = """You are an expert researcher. Conduct thorough research, and then write a polished report.

You have access to a few tools:
- internet_search: Run a web search for a given query.
"""

# Create and invoke the agent
agent = create_deep_agent(
    [internet_search],
    research_instructions,
)

result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
```

For a more elaborate demonstration, see: [examples/research/research_agent.py](examples/research/research_agent.py).

## Customizing Your Deep Agent

Customize your deep agent using the `create_deep_agent` function with these parameters:

### `tools` (Required)

Provide a list of functions or LangChain `@tool` objects for the agent to utilize.

### `instructions` (Required)

Define the agent's persona and goals with specific instructions.  This is combined with a built-in system prompt for optimal performance.

### `subagents` (Optional)

Integrate custom subagents to specialize in particular tasks.

```python
from deepagents import create_deep_agent

research_subagent = {
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

Select a specific [LangChain model object](https://python.langchain.com/docs/integrations/chat/) (defaults to `"claude-sonnet-4-20250514"`).

#### Example: Custom Model

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

### `builtin_tools` (Optional)

Control the available [built-in tools](#built-in-tools) by specifying which tools the agent should have access to.

```python
builtin_tools = ["write_todos"]
agent = create_deep_agent(..., builtin_tools=builtin_tools, ...)
```

## Deep Agent Components

### System Prompt

`deepagents` includes a carefully crafted [built-in system prompt](src/deepagents/prompts.py) inspired by Claude Code to guide the agent's behavior.

### Planning Tool

A built-in planning tool, based on ClaudeCode's TodoWrite, enables agents to strategize and break down complex tasks.

### File System Tools

Simulated file system tools (`ls`, `edit_file`, `read_file`, `write_file`) for interacting with a virtual file system using LangGraph's State object.

### Sub Agents

Utilize subagents for context quarantine and task specialization. A general-purpose subagent is always available. You can also define [custom sub agents](#subagents-optional).

### Built-in Tools

The default set of tools includes:

*   `write_todos`
*   `write_file`
*   `read_file`
*   `ls`
*   `edit_file`

### Human-in-the-Loop

Implement human oversight for tool execution using the `interrupt_config` parameter.

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

#### Approve

```python
config = {"configurable": {"thread_id": "1"}}
for s in agent.stream({"messages": [{"role": "user", "content": message}]}, config=config):
    print(s)
# If this calls a tool with an interrupt, this will then return an interrupt
for s in agent.stream(Command(resume=[{"type": "accept"}]), config=config):
    print(s)
```

#### Edit

```python
config = {"configurable": {"thread_id": "1"}}
for s in agent.stream({"messages": [{"role": "user", "content": message}]}, config=config):
    print(s)
# If this calls a tool with an interrupt, this will then return an interrupt
# Replace the `...` with the tool name you want to call, and the arguments
for s in agent.stream(Command(resume=[{"type": "edit", "args": {"action": "...", "args": {...}}}]), config=config):
    print(s)
```

#### Respond

```python
config = {"configurable": {"thread_id": "1"}}
for s in agent.stream({"messages": [{"role": "user", "content": message}]}, config=config):
    print(s)
# If this calls a tool with an interrupt, this will then return an interrupt
# Replace the `...` with the response to use all the ToolMessage content
for s in agent.stream(Command(resume=[{"type": "response", "args": "..."}]), config=config):
    print(s)
```

## MCP Integration

Integrate `deepagents` with MCP tools for a more extensive tool set.

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
*   \[ ] Code cleanup (type hinting, docstrings, formatting)
*   \[ ] Improve the virtual file system
*   \[ ] Create a deep coding agent example
*   \[ ] Benchmark the research agent example