# Build Powerful Multi-Agent Workflows with OpenAI Agents SDK

**The OpenAI Agents SDK empowers developers to create sophisticated, multi-agent workflows with ease.** ([Back to original repo](https://github.com/openai/openai-agents-python))

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

The OpenAI Agents SDK is a lightweight and versatile Python framework designed for building and orchestrating multi-agent workflows, offering flexibility and efficiency in LLM-powered applications.  It is provider-agnostic and supports various LLMs, including OpenAI models and 100+ others.

**Key Features:**

*   **Agents:** Configure LLMs with specific instructions, tools, and guardrails for targeted tasks.
*   **Handoffs:** Seamlessly transfer control between agents for complex workflows.
*   **Guardrails:** Implement configurable safety checks for input and output validation, ensuring reliability.
*   **Sessions:** Manage automatic conversation history across agent runs for context-aware interactions.
*   **Tracing:** Built-in tracing capabilities to view, debug, and optimize agent workflows.
*   **Extensible:** Integrate with external tracing processors for detailed analysis (Logfire, AgentOps, Braintrust, Scorecard, Keywords AI).
*   **Temporal Integration:** Use the SDK with [Temporal](https://temporal.io/) for long-running agents and human-in-the-loop tasks.

## Core Concepts

1.  [**Agents**](https://openai.github.io/openai-agents-python/agents): Define LLMs with specific roles, instructions, tools, and guardrails.
2.  [**Handoffs**](https://openai.github.io/openai-agents-python/handoffs/): Transfer control between agents during a workflow.
3.  [**Guardrails**](https://openai.github.io/openai-agents-python/guardrails/): Implement safety checks for inputs and outputs.
4.  [**Sessions**](#sessions): Manage conversation history across agent runs.
5.  [**Tracing**](https://openai.github.io/openai-agents-python/tracing/): Built-in tracking for debugging and optimization.

## Get Started

1.  **Set up your Python environment:**

    *   **Using `venv` (traditional):**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

    *   **Using `uv` (recommended):**

    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install the Agents SDK:**

    ```bash
    pip install openai-agents
    ```

    For voice support: `pip install 'openai-agents[voice]'`

## Hello World Example

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

**(Ensure you set the `OPENAI_API_KEY` environment variable)**

**(For Jupyter notebook users, see [hello_world_jupyter.ipynb](examples/basic/hello_world_jupyter.ipynb))**

## Handoffs Example

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

## Functions Example

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

The `Runner.run()` function orchestrates the agent loop until a final output is produced:

1.  Call the LLM with agent settings and message history.
2.  The LLM returns a response, potentially with tool calls.
3.  If a final output is returned, end the loop.
4.  If a handoff is indicated, switch to the new agent and go to step 1.
5.  Process any tool calls and append the responses, then go to step 1.

*   The `max_turns` parameter limits the loop's execution count.

### Final Output

Determined based on `output_type` or first response without tool calls/handoffs.

## Common Agent Patterns

The SDK supports diverse LLM workflow models, see examples in [`examples/agent_patterns`](examples/agent_patterns).

## Tracing

The Agents SDK has built-in tracing to monitor agent behavior.  It supports integration with external services (Logfire, AgentOps, Braintrust, Scorecard, Keywords AI).  Customize or disable tracing with the [Tracing documentation](http://openai.github.io/openai-agents-python/tracing).

## Long Running Agents & Human-in-the-Loop

Integrate the Agents SDK with [Temporal](https://temporal.io/) for durable workflows.  See [this video](https://www.youtube.com/watch?v=fFBZqzT4DD8) and [documentation here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).

## Sessions

Maintain conversation history automatically with built-in session memory.

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

*   **No memory** (default): Session parameter omitted.
*   **`session: Session = DatabaseSession(...)`**: Use a `Session` instance to manage history.

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

Create a class following the `Session` protocol.

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

## Development (For SDK/Example Edits)

0.  Ensure [`uv`](https://docs.astral.sh/uv/) is installed.

    ```bash
    uv --version
    ```

1.  Install dependencies

    ```bash
    make sync
    ```

2.  (After changes) lint/test

    ```
    make check # run tests linter and typechecker
    ```

    Individual commands:

    ```
    make tests  # run tests
    make mypy   # run typechecker
    make lint   # run linter
    make format-check # run style checker
    ```

## Acknowledgements

Thanks to the open-source community, especially:

*   [Pydantic](https://docs.pydantic.dev/latest/) and [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

We're committed to open-source development.