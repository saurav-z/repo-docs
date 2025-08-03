# Build Intelligent Workflows with the OpenAI Agents SDK

**Unlock the power of multi-agent workflows with the OpenAI Agents SDK, a lightweight and versatile Python framework designed for creating sophisticated AI applications.**  [View the original repository on GitHub](https://github.com/openai/openai-agents-python).

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

> [!NOTE]
> Looking for the JavaScript/TypeScript version? Check out [Agents SDK JS/TS](https://github.com/openai/openai-agents-js).

## Key Features

*   **Agent-Driven Workflows:**  Define and orchestrate LLMs (Large Language Models) as intelligent agents with specific instructions, tools, and handoffs.
*   **Provider-Agnostic:**  Seamlessly integrate with OpenAI's APIs, Chat Completions, and over 100+ other LLMs.
*   **Handoffs:** Enable efficient agent-to-agent communication and workflow transitions.
*   **Guardrails:** Implement configurable safety checks for robust input and output validation, ensuring reliability.
*   **Session Management:** Leverage automatic conversation history management for persistent and context-aware agent interactions.
*   **Built-in Tracing:**  Monitor, debug, and optimize your agent workflows with built-in tracing capabilities, which can be integrated with various external services like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI.
*   **Temporal Integration:** Integrate long-running workflows and human-in-the-loop tasks with the [Temporal](https://temporal.io/) integration.

## Core Concepts

1.  **Agents:** LLMs configured with instructions, tools, guardrails, and handoffs ([Agents documentation](https://openai.github.io/openai-agents-python/agents)).
2.  **Handoffs:** Specialized tool calls to transfer control between agents ([Handoffs documentation](https://openai.github.io/openai-agents-python/handoffs/)).
3.  **Guardrails:** Configurable safety checks ([Guardrails documentation](https://openai.github.io/openai-agents-python/guardrails/)).
4.  **Sessions:** Automatic conversation history management.
5.  **Tracing:** Built-in tracking of agent runs ([Tracing documentation](https://openai.github.io/openai-agents-python/tracing/)).

## Getting Started

### Prerequisites

*   Python 3.8 or higher
*   An OpenAI API key (or access to another supported LLM)

### Installation

1.  **Set up your Python environment** (choose one):

    *   **Using venv (traditional):**
        ```bash
        python -m venv env
        source env/bin/activate  # On Windows: env\Scripts\activate
        ```
    *   **Using uv (recommended):**
        ```bash
        uv venv
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        ```

2.  **Install the Agents SDK:**

    ```bash
    pip install openai-agents
    ```

    *   For voice support: `pip install 'openai-agents[voice]'`

## Examples

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

## The Agent Loop

The `Runner.run()` method executes an agent loop until a final output is generated.

1.  The LLM is called with the agent's settings and message history.
2.  The LLM returns a response, which can include tool calls.
3.  If the response has a final output, it is returned, and the loop ends.
4.  If the response has a handoff, control is transferred to the new agent, and the loop restarts.
5.  Tool calls are processed (if any), and tool responses are added to the messages. Then the loop restarts.

*   You can limit the loop iterations with the `max_turns` parameter.

### Final Output

*   If an `output_type` is set, the final output is when the LLM returns data matching that type.
*   If no `output_type` is set, the first LLM response without tool calls or handoffs is considered the final output.

## Common Agent Patterns

The SDK is designed to support diverse LLM workflow patterns, including deterministic flows and iterative loops. Explore example patterns in [`examples/agent_patterns`](examples/agent_patterns).

## Tracing & Monitoring

Easily track and debug agent behavior with automatic tracing.  Integrate with external services for advanced monitoring (Logfire, AgentOps, Braintrust, Scorecard, Keywords AI).  See the [Tracing documentation](http://openai.github.io/openai-agents-python/tracing) for customization.

## Long Running Agents & Human-in-the-Loop

Integrate with [Temporal](https://temporal.io/) for durable, long-running workflows and human-in-the-loop tasks. See this [video](https://www.youtube.com/watch?v=fFBZqzT4DD8) for a demonstration.

## Sessions & Memory

Maintain conversation history across agent runs with built-in session memory.

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

Create a custom class following the `Session` protocol:

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

## Development

(Instructions for contributing to the SDK)

0.  Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.

```bash
uv --version
```

1.  Install dependencies

```bash
make sync
```

2.  (After making changes) lint/test

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

We acknowledge the contributions of the open-source community, including:

*   [Pydantic](https://docs.pydantic.dev/latest/) and [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

We are committed to developing the Agents SDK as an open-source framework to foster community expansion.