# Build Advanced AI Workflows with the OpenAI Agents SDK

**Unleash the power of multi-agent systems with the OpenAI Agents SDK, a versatile Python framework for orchestrating sophisticated LLM-powered workflows.** ([Original Repo](https://github.com/openai/openai-agents-python))

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

The OpenAI Agents SDK empowers developers to create complex AI applications by simplifying the creation, management, and coordination of multiple AI agents. It's designed to be provider-agnostic, supporting the OpenAI Responses and Chat Completions APIs, and 100+ other LLMs.

## Key Features:

*   **Multi-Agent Orchestration:** Design and manage interconnected agents for complex tasks.
*   **Handoffs:** Seamlessly transfer control between agents for flexible workflows.
*   **Guardrails:** Implement configurable safety checks for robust input/output validation.
*   **Sessions:** Maintain automatic conversation history for context-aware interactions.
*   **Tracing:** Built-in run tracking for debugging and optimization, with integrations for various tracing providers.
*   **Function Calling:** Leverage function calls within your agents for enhanced capabilities.
*   **Long Running Workflows:** Integrates with Temporal for durable workflows, including human-in-the-loop tasks.

## Core Concepts:

*   **Agents:**  LLMs configured with instructions, tools, guardrails, and handoffs. ([Documentation](https://openai.github.io/openai-agents-python/agents))
*   **Handoffs:** Special tool calls to transfer control between agents. ([Documentation](https://openai.github.io/openai-agents-python/handoffs/))
*   **Guardrails:** Safety checks for input and output validation. ([Documentation](https://openai.github.io/openai-agents-python/guardrails/))
*   **Sessions:** Automatic conversation history management across agent runs.
*   **Tracing:** Built-in tracking of agent runs, allowing you to view, debug and optimize your workflows. ([Documentation](https://openai.github.io/openai-agents-python/tracing/))

## Get Started

1.  **Set up your Python environment:**

    *   **Option A: Using venv**

        ```bash
        python -m venv env
        source env/bin/activate  # On Windows: env\Scripts\activate
        ```
    *   **Option B: Using uv (recommended)**

        ```bash
        uv venv
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        ```

2.  **Install the Agents SDK:**

    ```bash
    pip install openai-agents
    ```

    For voice support:
    ```bash
    pip install 'openai-agents[voice]'
    ```

## Example: Hello World

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

## Example: Handoffs

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

## Example: Functions

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

## Agent Loop

The agent loop continuously runs until a final output is generated.  Here's how it works:

1.  The LLM is called with agent settings and message history.
2.  The LLM response is checked for a final output (defined by `output_type`) or tool calls/handoffs.
3.  If a final output is present, it's returned and the loop ends.
4.  If a handoff is present, the agent is set to the new agent.
5.  Tool calls are processed and responses are added. Then the loop goes back to step 1.

A `max_turns` parameter limits the number of loop iterations.

### Final Output

*   If an `output_type` is set, the loop ends when the agent generates a structured output of that type.
*   If no `output_type` is set, the loop ends when the agent produces a response without any tool calls or handoffs.

## Common Agent Patterns

The SDK is designed for flexibility to create diverse LLM workflows, including deterministic and iterative flows.  Explore examples in the `examples/agent_patterns` directory.

## Tracing & Monitoring

The Agents SDK provides automatic tracing to track and debug agent behavior. Integrations are available for services like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI. Customize or disable tracing as needed.  See [Tracing](http://openai.github.io/openai-agents-python/tracing) for more information, including [external tracing processors](http://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list).

## Long Running Agents & Human-in-the-Loop

Use the Agents SDK's [Temporal](https://temporal.io/) integration for long-running workflows, including human-in-the-loop tasks.  View a demo of the Temporal/Agents SDK integration [in this video](https://www.youtube.com/watch?v=fFBZqzT4DD8), and [view docs here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).

## Sessions for Context

The SDK offers built-in session memory to preserve context across agent runs, eliminating the need to manage `.to_input_list()` manually.

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

*   **No memory** (default):  No session memory when the session parameter is omitted.
*   **`session: Session = DatabaseSession(...)`**:  Use a `Session` instance to manage conversation history.

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

Implement your own session memory by creating a class following the `Session` protocol:

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

## Development (for SDK/example modifications)

0.  Ensure [`uv`](https://docs.astral.sh/uv/) is installed.

    ```bash
    uv --version
    ```

1.  Install dependencies

    ```bash
    make sync
    ```

2.  Lint/test (after changes)

    ```bash
    make check # run tests linter and typechecker
    ```

    Or run individually:

    ```bash
    make tests  # run tests
    make mypy   # run typechecker
    make lint   # run linter
    make format-check # run style checker
    ```

## Acknowledgements

The Agents SDK builds on the work of the open-source community, with special thanks to:

*   [Pydantic](https://docs.pydantic.dev/latest/) and [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

We're committed to building the Agents SDK as an open source framework.