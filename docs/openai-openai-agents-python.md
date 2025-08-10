# Build Powerful Multi-Agent Workflows with OpenAI Agents SDK

**The OpenAI Agents SDK is your key to building intelligent, provider-agnostic multi-agent workflows with LLMs, enabling complex automation and streamlined interactions. ([Original Repo](https://github.com/openai/openai-agents-python))**

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

**Key Features:**

*   **Agent Orchestration:** Design and manage complex workflows by connecting multiple AI agents.
*   **Provider-Agnostic:** Supports OpenAI's API, Chat Completions, and 100+ other LLMs.
*   **Handoffs:** Seamlessly transfer control between agents for dynamic workflows.
*   **Guardrails:** Ensure safety and reliability with configurable input/output validation.
*   **Sessions:** Automatically manage conversation history for coherent interactions.
*   **Tracing:** Built-in tracing tools for easy debugging, optimization, and integration with external services (Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI).
*   **Functions:** Integrate external tools and APIs with function calling.
*   **Long Running Tasks:** Integration with Temporal for long-running workflows and human-in-the-loop tasks.

## Core Concepts

Explore the fundamental building blocks of the Agents SDK:

1.  **Agents:** LLMs configured with instructions, tools, guardrails, and handoffs. [Learn more](https://openai.github.io/openai-agents-python/agents)
2.  **Handoffs:** Special tool calls that facilitate agent control transfer. [Learn more](https://openai.github.io/openai-agents-python/handoffs/)
3.  **Guardrails:** Customizable safety checks for input and output validation.
4.  **Sessions:** Manage conversation history automatically across agent runs.
5.  **Tracing:** Built-in tracking and debugging for your workflows. [Learn more](https://openai.github.io/openai-agents-python/tracing/)

## Getting Started

Follow these steps to set up your environment and start using the Agents SDK:

1.  **Set up your Python environment:**

    *   **Option A: Using venv:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

    *   **Option B: Using uv (recommended):**
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
2.  **Install the Agents SDK:**
    ```bash
    pip install openai-agents
    ```
    For voice support, install with the optional `voice` group: `pip install 'openai-agents[voice]'`.

## Examples

Get hands-on experience with the SDK through these examples:

### Hello World

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```
(_Ensure you set the `OPENAI_API_KEY` environment variable_)
(_For Jupyter notebook users, see [hello_world_jupyter.ipynb](examples/basic/hello_world_jupyter.ipynb)_)

### Handoffs

```python
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)


async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)
    # ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?


if __name__ == "__main__":
    asyncio.run(main())
```

### Functions

```python
import asyncio

from agents import Agent, Runner, function_tool


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)


async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())
```

## Agent Loop and Output

Understand the execution flow and output behavior of agents:

*   The `Runner.run()` function orchestrates a loop until a final output is generated.
*   The agent interacts with the LLM, processes responses, and executes tool calls or handoffs.
*   Final output is determined by `output_type` or the absence of tool calls/handoffs in the LLM's response.

## Common Agent Patterns

The Agents SDK enables flexible workflow designs, from simple to complex, including deterministic flows and iterative loops. Explore examples in the [`examples/agent_patterns`](examples/agent_patterns) directory.

## Tracing for Debugging and Optimization

The Agents SDK's built-in tracing simplifies tracking and debugging. You can extend tracing with custom spans and integrate with external services like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI.  Refer to [Tracing Documentation](http://openai.github.io/openai-agents-python/tracing/) for customization.

## Long Running Agents & Human-in-the-Loop

Integrate with [Temporal](https://temporal.io/) to create durable, long-running workflows, including human-in-the-loop tasks.  See the Agents SDK and Temporal working together in this [video](https://www.youtube.com/watch?v=fFBZqzT4DD8), and [view docs here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).

## Sessions for Persistent Context

The SDK includes built-in session memory to maintain conversation history across multiple agent runs.

### Quick Start

```python
from agents import Agent, Runner, SQLiteSession

# Create agent
agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

# Create a session instance
session = SQLiteSession("conversation_123")

# First turn
result = await Runner.run(
    agent,
    "What city is the Golden Gate Bridge in?",
    session=session
)
print(result.final_output)  # "San Francisco"

# Second turn - agent automatically remembers previous context
result = await Runner.run(
    agent,
    "What state is it in?",
    session=session
)
print(result.final_output)  # "California"

# Also works with synchronous runner
result = Runner.run_sync(
    agent,
    "What's the population?",
    session=session
)
print(result.final_output)  # "Approximately 39 million"
```

### Session Options

*   **No memory** (default): No session memory when session parameter is omitted
*   **`session: Session = DatabaseSession(...)`**: Use a Session instance to manage conversation history

```python
from agents import Agent, Runner, SQLiteSession

# Custom SQLite database file
session = SQLiteSession("user_123", "conversations.db")
agent = Agent(name="Assistant")

# Different session IDs maintain separate conversation histories
result1 = await Runner.run(
    agent,
    "Hello",
    session=session
)
result2 = await Runner.run(
    agent,
    "Hello",
    session=SQLiteSession("user_456", "conversations.db")
)
```

### Custom Session Implementations

Implement your own session memory using the `Session` protocol:

```python
from agents.memory import Session
from typing import List

class MyCustomSession:
    """Custom session implementation following the Session protocol."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        # Your initialization here

    async def get_items(self, limit: int | None = None) -> List[dict]:
        # Retrieve conversation history for the session
        pass

    async def add_items(self, items: List[dict]) -> None:
        # Store new items for the session
        pass

    async def pop_item(self) -> dict | None:
        # Remove and return the most recent item from the session
        pass

    async def clear_session(self) -> None:
        # Clear all items for the session
        pass

# Use your custom session
agent = Agent(name="Assistant")
result = await Runner.run(
    agent,
    "Hello",
    session=MyCustomSession("my_session")
)
```

## Development (for contributing)

These instructions are for those wishing to modify the SDK or its examples.

0. Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.
```bash
uv --version
```
1. Install dependencies
```bash
make sync
```
2. Run checks after changes
```
make check # run tests linter and typechecker
```
Or run them individually:
```
make tests  # run tests
make mypy   # run typechecker
make lint   # run linter
make format-check # run style checker
```

## Acknowledgements

The OpenAI Agents SDK leverages the contributions of the open-source community, with special thanks to:

*   [Pydantic](https://docs.pydantic.dev/latest/) & [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

The OpenAI Agents SDK is an open-source project designed to foster community-driven development.