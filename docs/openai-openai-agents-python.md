# Build Powerful LLM Workflows with the OpenAI Agents SDK

**Empower your applications with intelligent, multi-agent workflows using the OpenAI Agents SDK, a flexible and provider-agnostic framework.** Explore the original repository [here](https://github.com/openai/openai-agents-python).

## Key Features

*   **Agent-Based Architecture:** Define and orchestrate intelligent agents with instructions, tools, guardrails, and handoffs.
*   **Provider-Agnostic:** Seamlessly integrate with OpenAI's APIs, as well as 100+ other LLMs.
*   **Handoffs:** Enable dynamic control transfer between agents for complex task execution.
*   **Guardrails:** Implement configurable safety checks for robust input and output validation.
*   **Sessions:** Maintain persistent conversation history across agent runs for context-aware interactions.
*   **Tracing:** Built-in tracing for easy debugging, optimization, and integration with external platforms like Logfire, AgentOps, and more.
*   **Long-Running Workflows:** Integrate with Temporal for durable, human-in-the-loop tasks.
*   **Function Calling:** Enables agents to call functions to interact with the outside world, such as calling weather APIs.

## Get Started

### Prerequisites:

*   Python 3.9 or newer
*   An OpenAI API key

### Installation

1.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
2.  **Install the OpenAI Agents SDK:**

    ```bash
    pip install openai-agents
    ```
3.  **For voice support (optional):**

    ```bash
    pip install 'openai-agents[voice]'
    ```

### Alternative installation with `uv`:

If you're familiar with [uv](https://docs.astral.sh/uv/), using the tool would be even similar:

```bash
uv init
uv add openai-agents
```

For voice support, install with the optional `voice` group: `uv add 'openai-agents[voice]'`.

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

**Note:**  Make sure you have set the `OPENAI_API_KEY` environment variable. For a Jupyter Notebook example, see `examples/basic/hello_world_jupyter.ipynb`.

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

## Core Concepts

*   **Agents:** LLMs configured with instructions, tools, guardrails, and handoffs.
*   **Handoffs:** Enables transferring control between agents, which enables creating more complex workflows.
*   **Guardrails:** Configurable safety checks for input and output validation.
*   **Sessions:** Automatically manages conversation history across agent runs, which helps you make your agents context-aware.
*   **Tracing:** Built-in tracing of agent runs, allowing you to view, debug and optimize your workflows.

## The Agent Loop

The agent loop is what handles the execution of the agent. The loop works by:

1.  Calling the LLM, using the model and settings on the agent, and the message history.
2.  The LLM returns a response, which may include tool calls.
3.  If the response has a final output, we return it and end the loop.
4.  If the response has a handoff, we set the agent to the new agent and go back to step 1.
5.  We process the tool calls (if any) and append the tool responses messages. Then we go to step 1.

There is a `max_turns` parameter that you can use to limit the number of times the loop executes.

### Final Output

The agent loop runs until a final output is generated. This output is determined by either:

1.  If you set an `output_type` on the agent, the final output is when the LLM returns something of that type. We use [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) for this.
2.  If there's no `output_type` (i.e. plain text responses), then the first LLM response without any tool calls or handoffs is considered as the final output.

## Common Agent Patterns

The Agents SDK is designed to be highly flexible, allowing you to model a wide range of LLM workflows including deterministic flows, iterative loops, and more. See examples in [`examples/agent_patterns`](examples/agent_patterns).

## Tracing

The Agents SDK automatically traces your agent runs, making it easy to track and debug the behavior of your agents. Tracing is extensible by design, supporting custom spans and a wide variety of external destinations, including [Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents), [AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk), [Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk), [Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration), and [Keywords AI](https://docs.keywordsai.co/integration/development-frameworks/openai-agent). For more details about how to customize or disable tracing, see [Tracing](http://openai.github.io/openai-agents-python/tracing), which also includes a larger list of [external tracing processors](http://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list).

## Long Running Agents & Human-in-the-Loop

You can use the Agents SDK [Temporal](https://temporal.io/) integration to run durable, long-running workflows, including human-in-the-loop tasks. View a demo of Temporal and the Agents SDK working in action to complete long-running tasks [in this video](https://www.youtube.com/watch?v=fFBZqzT4DD8), and [view docs here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).

## Sessions

The Agents SDK provides built-in session memory to automatically maintain conversation history across multiple agent runs, eliminating the need to manually handle `.to_input_list()` between turns.

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

You can implement your own session memory by creating a class that follows the `Session` protocol:

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

## Development (for SDK contributors)

1.  Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.

    ```bash
    uv --version
    ```

2.  Install dependencies

    ```bash
    make sync
    ```

3.  (After making changes) lint/test

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

The OpenAI Agents SDK leverages the power of the open-source community, including:

*   [Pydantic](https://docs.pydantic.dev/latest/) and [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

The development of this SDK is driven by a commitment to open source and community collaboration.