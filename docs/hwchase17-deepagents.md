# DeepAgents: Build Powerful, Deep-Thinking AI Agents (Python)

**Tired of shallow AI agents?** DeepAgents empowers you to create sophisticated, autonomous AI systems capable of complex tasks through planning, sub-agents, and advanced tools. ➡️ [Check out the original repo!](https://github.com/hwchase17/deepagents)

<img src="deep_agents.png" alt="deep agent" width="600"/>

DeepAgents provides a Python package for building advanced agents that can perform complex tasks using Large Language Models (LLMs). Inspired by architectures like "Claude Code," "Deep Research," and "Manus," it offers a flexible framework for creating agents that can plan, use tools, and manage sub-agents, including:

**Key Features:**

*   **Planning Tool:**  A built-in planning mechanism to enable the agent to strategize and break down complex tasks.
*   **Sub-Agents:**  Easily manage and integrate sub-agents for modularity and context isolation.
*   **File System Tools:** Built-in mock file system tools (`ls`, `edit_file`, `read_file`, `write_file`) for file interactions within the agent's context (no persistent storage).
*   **Customizable Prompts:** Fine-tune agent behavior with custom instructions and a powerful built-in system prompt.
*   **Tool Interrupts:** Implement human-in-the-loop tool approval with configurable settings for added safety and control.
*   **LangGraph Integration:**  Built on LangGraph, giving you access to streaming, memory, and other LangGraph features.
*   **MCP Support:** Run agents with MCP tools through Langchain MCP Adapter library

## Installation

```bash
pip install deepagents
```

## Quickstart: Research Agent Example

Get started with a simple research agent:

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

## Customizing Deep Agents

The `create_deep_agent` function accepts the following parameters for customization:

### `tools` (Required)

A list of functions or LangChain `@tool` objects that the agent and sub-agents can utilize.

### `instructions` (Required)

The instructions that the agent will follow to guide its behavior, which are integrated within a system prompt.

### `subagents` (Optional)

Define and incorporate custom sub-agents, each with their own instructions, tools, and context to handle specific tasks.  This promotes context isolation and modularity.

```python
# Example using a custom subagent
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

Customize the LLM used by the agent.  Defaults to `"claude-sonnet-4-20250514"`.

#### Example: Custom Model

```python
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama

model = ChatOllama(model="gpt-oss:20b")  # Example using Ollama
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
)
```

#### Example: Per-subagent model override

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

*   **System Prompt:** A built-in, comprehensive prompt that guides the agent's behavior, inspired by Claude Code.  It includes instructions for using the planning tool, file system tools, and sub-agents.
*   **Planning Tool:**  A basic planning tool to help the agent strategize and manage tasks.
*   **File System Tools:** Mock file system tools for basic file operations within the agent's context.
*   **Sub Agents:**  Easily integrate sub-agents for specialized tasks and better context management.
*   **Tool Interrupts:** Implement human-in-the-loop tool approval for safe tool execution.

## Roadmap

*   \[ ] Allow users to customize full system prompt
*   \[ ] Code cleanliness (type hinting, docstrings, formating)
*   \[ ] Allow for more of a robust virtual filesystem
*   \[ ] Create an example of a deep coding agent built on top of this
*   \[ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)