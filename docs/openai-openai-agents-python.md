# Build Advanced Multi-Agent Workflows with the OpenAI Agents SDK

**Unleash the power of AI by building sophisticated multi-agent systems for complex tasks with the OpenAI Agents SDK.**  [View the original repository](https://github.com/openai/openai-agents-python).

![Image of the Agents Tracing UI](https://cdn.openai.com/API/docs/images/orchestration.png)

> **Note:** Looking for the JavaScript/TypeScript version? Check out [Agents SDK JS/TS](https://github.com/openai/openai-agents-js).

## Key Features

*   **Agent Orchestration:** Design workflows using independent agents with specific roles, instructions, and access to tools and other agents.
*   **Flexible LLM Support:** Integrate with OpenAI's APIs, as well as 100+ other Large Language Models via a provider-agnostic architecture.
*   **Agent Handoffs:** Seamlessly transfer control between agents for efficient task management.
*   **Guardrails for Safety:** Configure input and output validation to ensure reliable and secure agent interactions.
*   **Session Management:** Automate conversation history tracking across agent runs with built-in session memory.
*   **Built-in Tracing & Debugging:** Leverage built-in tracing capabilities to visualize and analyze agent behavior. Support for integrations with Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI.
*   **Long-Running Workflows:** Integrate with [Temporal](https://temporal.io/) to run durable, long-running workflows, including human-in-the-loop tasks.

## Core Concepts

*   **Agents:** LLMs configured with instructions, tools, guardrails, and handoffs.
*   **Handoffs:** A specialized tool call used by the Agents SDK for transferring control between agents.
*   **Guardrails:** Configurable safety checks for input and output validation.
*   **Sessions:** Automatic conversation history management across agent runs.
*   **Tracing:** Built-in tracking of agent runs, allowing you to view, debug and optimize your workflows.

Learn more in the [documentation](https://openai.github.io/openai-agents-python/).

## Get Started

1.  **Set up your Python environment**

    *   **Option A: Using venv (traditional method)**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

    *   **Option B: Using uv (recommended)**

    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install Agents SDK**

    ```bash
    pip install openai-agents
    ```

    For voice support, install with the optional `voice` group: `pip install 'openai-agents[voice]'`.

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

(_If running this, ensure you set the `OPENAI_API_KEY` environment variable_)

(_For Jupyter notebook users, see [hello_world_jupyter.ipynb](examples/basic/hello_world_jupyter.ipynb)_)

### Handoffs Example

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

### Functions Example

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

## Agent Loop & Final Output

The `Runner.run()` function runs a loop, calling the LLM, handling responses, and managing tool calls and handoffs. The loop continues until a `final_output` is produced, as determined by either:

1.  If an `output_type` is set on the agent, the final output is when the LLM returns something of that type. We use [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) for this.
2.  If there's no `output_type` (i.e. plain text responses), then the first LLM response without any tool calls or handoffs is considered as the final output.

## Common Agent Patterns

Explore the [`examples/agent_patterns`](examples/agent_patterns) directory for flexible and adaptable patterns for various LLM workflows, including deterministic and iterative flows.

## Tracing

The Agents SDK automatically traces agent runs to easily track and debug agent behavior. It supports a variety of external destinations, including [Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents), [AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk), [Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk), [Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration), and [Keywords AI](https://docs.keywordsai.co/integration/development-frameworks/openai-agent).  See the [Tracing documentation](http://openai.github.io/openai-agents-python/tracing/) for more details.

## Long Running Agents & Human-in-the-Loop

Use the Agents SDK [Temporal](https://temporal.io/) integration to build robust, long-running workflows. Watch a demo [in this video](https://www.youtube.com/watch?v=fFBZqzT4DD8), and [view docs here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).

## Sessions

The Agents SDK includes built-in session memory to maintain conversation history across runs.

### Quick start

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

### Session options

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

### Custom session implementations

You can create custom session memory by implementing the `Session` protocol:

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

(Only needed if you need to edit the SDK/examples)

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

We acknowledge the open-source community, especially:

*   [Pydantic](https://docs.pydantic.dev/latest/) and [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

We are dedicated to developing the Agents SDK as an open source framework.