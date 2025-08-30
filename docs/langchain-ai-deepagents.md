# DeepAgents: Build Powerful, Multi-Step AI Agents (Powered by LangGraph)

**Unlock the power of complex AI agents with DeepAgents, a Python package built for creating sophisticated, multi-step agents that can plan, research, and execute intricate tasks.**  [See the original repository](https://github.com/langchain-ai/deepagents) for more information.

<img src="deep_agents.png" alt="deep agent" width="600"/>

DeepAgents empowers you to easily build "deep" agents capable of handling complex tasks, drawing inspiration from cutting-edge applications like "Claude Code". It leverages the key ingredients for advanced agent behavior: planning, sub-agents, virtual file systems, and detailed prompting.

## Key Features:

*   **Simplified Agent Creation:** Easily create custom deep agents using the `create_deep_agent` function.
*   **Built-in Planning Tool:** Includes a planning tool inspired by Claude Code's TodoWrite tool to help agents strategize.
*   **Virtual File System:** Provides built-in file system tools (read, write, edit, list) using LangGraph's State object for persistent storage.
*   **Sub-Agent Support:** Enables the use of sub-agents for context quarantine and specialized tasks, inspired by Claude Code.
*   **Customizable Prompts:** Leverage a default system prompt, inspired by Claude Code, while also allowing you to customize it.
*   **Human-in-the-Loop Tool Interrupts:**  Integrate human approval for tool execution using the `interrupt_config` parameter to ensure that tool usage happens as you specify.
*   **LangGraph Integration:** Built on LangGraph for flexibility with streaming, human-in-the-loop interactions, memory, and studio integration.
*   **MCP Support:** Integrate tools with MultiServerMCPClient.

## Installation

```bash
pip install deepagents
```

## Usage Examples

Here's a basic example to get you started (requires `pip install tavily-python`):

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

For a more advanced research agent example, explore [examples/research/research_agent.py](examples/research/research_agent.py).

## Customization Options

### Tools (Required)

Pass a list of functions or LangChain `@tool` objects to the `tools` parameter to give the agent access to various tools.

### Instructions (Required)

Define the agent's role and behavior through the `instructions` parameter. This is part of the prompt the agent receives.

### Subagents (Optional)

Create specialized sub-agents with their own instructions and toolsets for context quarantine or focused tasks. Define sub-agents using the following schema:

```python
class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    model_settings: NotRequired[dict[str, Any]]
```

### Model (Optional)

Specify a custom LangChain model with the `model` parameter, such as:
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
### Built-in Tools (Optional)

Customize the default built-in tools with the `builtin_tools` parameter. By default, agents have access to: `write_todos`, `write_file`, `read_file`, `ls`, and `edit_file`. You can also disable these.

### Tool Interrupts

Use `interrupt_config` to enable human-in-the-loop approval for tool execution.  Configure tool-specific settings with the following parameters:
*   `allow_ignore`: Allows users to skip the tool call.
*   `allow_respond`: Permits users to add a text response.
*   `allow_edit`: Enables users to modify tool arguments.
*   `allow_accept`: Allows users to approve the tool call.

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

## Deep Agent Architecture Details

*   **System Prompt:** The built-in prompt provides the agent with detailed instructions for using planning, file system tools, and sub-agents, inspired by Claude Code.
*   **Planning Tool:** Enables agents to create plans, guiding their actions.
*   **File System Tools:** Built-in tools facilitate virtual file operations using LangGraph's State object.
*   **Sub-Agents:** Leverage sub-agents for context management and specialized task execution.
*   **Built-in Tools:** Offers five default tools for common operations.
*   **Tool Interrupts:** Implement human oversight for controlled tool execution.

## Roadmap

*   [ ] Allow users to customize full system prompt
*   [ ] Code cleanliness (type hinting, docstrings, formating)
*   [ ] Allow for more of a robust virtual filesystem
*   [ ] Create an example of a deep coding agent built on top of this
*   [ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)