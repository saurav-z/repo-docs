# 🧠 DeepAgents: Build Powerful, "Deep" LLM Agents

**Tackle complex tasks with ease by creating sophisticated, multi-step LLM agents using DeepAgents. Inspired by Claude Code, this library provides the building blocks for agents that plan, use tools, and handle complex instructions.**

[View the original repository](https://github.com/hwchase17/deepagents)

**Key Features:**

*   **Simplified Agent Creation:** Easily create deep agents with a focus on planning, tool use, and complex task execution.
*   **Built-in Planning Tool:** Implement a "todo" planning tool for agents to come up with a plan.
*   **Virtual File System:** Interact with a mock file system for persistent storage and retrieval using `ls`, `edit_file`, `read_file`, `write_file` tools.
*   **Sub-Agent Support:** Delegate tasks and manage context with easily configurable sub-agents.
*   **Human-in-the-Loop:** Approve, edit, or respond to tool calls for safety and control.
*   **Customizable:** Tailor your agents with custom tools, instructions, and model settings, including advanced features like per-subagent model overrides.
*   **Integration:** Works seamlessly with LangGraph, allowing for streaming, human-in-the-loop interaction, memory, and studio integration.
*   **Async Support:** Supports asynchronous tools.
*   **MCP Support:** Integrates with MCP tools.
*   **Configurable Agent:** Allows you to configure the agent via a config passed in.

## Installation

```bash
pip install deepagents
```

## Usage Example: Research Agent

**(Requires `pip install tavily-python`)**

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

## Customizing Your Deep Agents

`deepagents` offers extensive customization options for creating powerful, specialized agents:

### `tools` (Required)

Pass a list of functions or LangChain `@tool` objects for your agent to use.

### `instructions` (Required)

Provide a prompt to guide your agent's behavior.  A built-in system prompt is also used.

### `subagents` (Optional)

Define and configure sub-agents for focused tasks. Each subagent has its own `name`, `description`, `prompt`, optional `tools` and optional `model_settings` to perform specific functions.

```python
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

Customize the LLM used by your agent by passing a [LangChain model object](https://python.langchain.com/docs/integrations/chat/).  Defaults to `"claude-sonnet-4-20250514"`.

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

### `builtin_tools` (Optional)

Control which built-in tools are available to your agent by specifying their names (e.g., `["write_todos"]`).

## Deep Agent Components

*   **System Prompt:** Built-in prompt based on Claude Code's prompt provides essential instructions.
*   **Planning Tool:** Uses a "todo" list to help organize the agent's actions.
*   **File System Tools:**  Includes `ls`, `edit_file`, `read_file`, and `write_file` for simulated file operations within the agent's context.
*   **Sub Agents:** Allows for calling subagents with their own specialized prompts and instructions.  Includes a `general-purpose` subagent by default.
*   **Built-In Tools:**  Includes `write_todos`, `write_file`, `read_file`, `ls`, and `edit_file` by default, with options to disable.
*   **Human-in-the-Loop:** Supports human review for tool execution using the `interrupt_config` parameter.

## Human-in-the-Loop

`deepagents` supports human-in-the-loop approval for tool execution. You can configure specific tools to require human approval before execution using the `interrupt_config` parameter.

To use human in the loop, you need to have a checkpointer attached.
Note: if you are using LangGraph Platform, this is automatically attached.

Example usage:

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

To "approve" a tool call means the agent will execute the tool call as is.

This flow shows how to approve a tool call (assuming the tool requiring approval is called):

```python
config = {"configurable": {"thread_id": "1"}}
for s in agent.stream({"messages": [{"role": "user", "content": message}]}, config=config):
    print(s)
# If this calls a tool with an interrupt, this will then return an interrupt
for s in agent.stream(Command(resume=[{"type": "accept"}]), config=config):
    print(s)

```

#### Edit

To "edit" a tool call means the agent will execute the new tool with the new arguments. You can change both the tool to call or the arguments to pass to that tool.

The `args` parameter you pass back should be a dictionary with two keys:

- `action`: maps to a string which is the name of the tool to call
- `args`: maps to a dictionary which is the arguments to pass to the tool

This flow shows how to edit a tool call (assuming the tool requiring approval is called):

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

To "respond" to a tool call means that tool is NOT called. Rather, a tool message is appended with the content you respond with, and the updated messages list is then sent back to the model.

The `args` parameter you pass back should be a string with your response.

This flow shows how to respond to a tool call (assuming the tool requiring approval is called):

```python
config = {"configurable": {"thread_id": "1"}}
for s in agent.stream({"messages": [{"role": "user", "content": message}]}, config=config):
    print(s)
# If this calls a tool with an interrupt, this will then return an interrupt
# Replace the `...` with the response to use all the ToolMessage content
for s in agent.stream(Command(resume=[{"type": "response", "args": "..."}]), config=config):
    print(s)

```
## Async

If you are passing async tools to your agent, you will want to `from deepagents import async_create_deep_agent`

## MCP

The `deepagents` library can be ran with MCP tools. This can be achieved by using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

**NOTE:** will want to use `from deepagents import async_create_deep_agent` to use the async version of `deepagents`, since MCP tools are async

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
    agent = async_create_deep_agent(tools=mcp_tools, ....)

    # Stream the agent
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is langgraph?"}]},
        stream_mode="values"
    ):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()

asyncio.run(main())
```

## Configurable Agent

Configurable agents allow you to control the agent via a config passed in.

```python
from deepagents import create_configurable_agent

agent_config = {"instructions": "foo", "subagents": []}

build_agent = create_configurable_agent(
    agent_config['instructions'],
    agent_config['subagents'],
    [],
    agent_config={"recursion_limit": 1000}
)
```
You can now use `build_agent` in your `langgraph.json` and deploy it with `langgraph dev`

For async tools, you can use `from deepagents import async_create_configurable_agent`

## Roadmap

*   \[ ] Allow users to customize full system prompt
*   \[ ] Code cleanliness (type hinting, docstrings, formating)
*   \[ ] Allow for more of a robust virtual filesystem
*   \[ ] Create an example of a deep coding agent built on top of this
*   \[ ] Benchmark the example of \[deep research agent](examples/research/research_agent.py)