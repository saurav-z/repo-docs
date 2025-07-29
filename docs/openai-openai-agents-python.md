# Build Powerful Multi-Agent Workflows with the OpenAI Agents SDK

**Unlock the power of collaborative AI by effortlessly creating and managing multi-agent workflows with the OpenAI Agents SDK, supporting a wide range of LLMs and providing built-in features for seamless orchestration.**  ([View the original repository](https://github.com/openai/openai-agents-python))

The OpenAI Agents SDK is a powerful Python framework for building multi-agent systems. It's designed to be flexible and easy to use, enabling you to create complex workflows with LLMs, tools, and more.

Key Features:

*   **Agent Orchestration:**  Easily define and connect multiple agents to create sophisticated workflows.
*   **Provider Agnostic:** Compatible with OpenAI's APIs, chat completions, and 100+ other LLMs through a unified interface.
*   **Handoffs:** Enables seamless agent-to-agent control transfer.
*   **Guardrails:** Configurable safety checks for input and output validation.
*   **Sessions:** Automatic conversation history management across agent runs.
*   **Tracing:** Built-in tracing capabilities for monitoring, debugging, and optimizing your agent workflows, with support for external integrations (Logfire, AgentOps, Braintrust, Scorecard, Keywords AI, and more).

## Getting Started

Follow these steps to set up your environment and install the SDK:

1.  **Set up your Python Environment:**
    *   **Option A: Using `venv` (traditional method):**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

    *   **Option B: Using `uv` (recommended):**

    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install the Agents SDK:**

    ```bash
    pip install openai-agents
    ```

    *   For voice support: `pip install 'openai-agents[voice]'`

## Quick Examples

### Hello World

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
```

(_Ensure you set the `OPENAI_API_KEY` environment variable_)

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

### Function Calling

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

## Core Concepts

*   [**Agents**](https://openai.github.io/openai-agents-python/agents): LLMs with instructions, tools, guardrails, and handoffs.
*   [**Handoffs**](https://openai.github.io/openai-agents-python/handoffs/): Tool calls to transfer control between agents.
*   [**Guardrails**](https://openai.github.io/openai-agents-python/guardrails/): Input/output validation for safety.
*   **Sessions:** Manage conversation history across agent runs.  See [Sessions](#sessions) below.
*   [**Tracing**](https://openai.github.io/openai-agents-python/tracing/): Monitor, debug, and optimize agent workflows.

### The Agent Loop

The `Runner.run()` method executes the following loop:

1.  Call the LLM.
2.  Process the LLM response.
3.  If a final output is found, return it.
4.  If a handoff is present, switch to the new agent.
5.  Process tool calls and respond to the tools (if any).
6.  Go back to step 1.

### Final Output

*   If an `output_type` is set, the final output matches that type.
*   If no `output_type`, the first LLM response without tool calls or handoffs is considered the final output.

## Sessions for Contextual Conversations

The SDK offers built-in session memory to maintain conversation history automatically.

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

*   **No memory** (default): Omit the `session` parameter.
*   **`session: Session = DatabaseSession(...)`**: Use a Session instance.

### Custom Session Implementations

Create a class implementing the `Session` protocol for custom session memory:

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

## Common Agent Patterns

The Agents SDK supports various LLM workflow designs, including deterministic flows and iterative loops.  See [`examples/agent_patterns`](examples/agent_patterns) for example implementations.

## Tracing

Trace and debug your agent workflows easily.  Tracing supports custom spans and external integrations (Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI, etc.). For customization or disabling tracing, see [Tracing](http://openai.github.io/openai-agents-python/tracing), which also includes a larger list of [external tracing processors](http://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list).

## Development

(For contributions)

0. Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.

```bash
uv --version
```

1. Install dependencies

```bash
make sync
```

2. (After making changes) lint/test

```
make check # run tests linter and typechecker
```

Or to run them individually:

```
make tests  # run tests
make mypy   # run typechecker
make lint   # run linter
make format-check # run style checker
```

## Acknowledgements

We'd like to acknowledge the excellent work of the open-source community, especially:

-   [Pydantic](https://docs.pydantic.dev/latest/) (data validation) and [PydanticAI](https://ai.pydantic.dev/) (advanced agent framework)
-   [LiteLLM](https://github.com/BerriAI/litellm) (unified interface for 100+ LLMs)
-   [MkDocs](https://github.com/squidfunk/mkdocs-material)
-   [Griffe](https://github.com/mkdocstrings/griffe)
-   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)