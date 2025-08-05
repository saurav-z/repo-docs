# Build Powerful Multi-Agent Workflows with the OpenAI Agents SDK

**The OpenAI Agents SDK empowers developers to effortlessly create sophisticated, multi-agent workflows, orchestrating LLMs for complex tasks.** [Check out the original repo on GitHub!](https://github.com/openai/openai-agents-python)

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

This Python SDK simplifies the development of AI-powered applications by offering a flexible and efficient framework.  It supports the OpenAI Responses and Chat Completions APIs, and is also provider-agnostic, working with 100+ other LLMs.

**Key Features:**

*   **Agents:** Define LLMs with instructions, tools, guardrails, and handoffs.
*   **Handoffs:** Seamlessly transfer control between agents for complex tasks.
*   **Guardrails:** Implement configurable safety checks for input and output validation to ensure responsible AI development.
*   **Sessions:** Automatically manage conversation history across agent runs for persistent context.
*   **Tracing:** Built-in tracing to monitor, debug, and optimize your agent workflows. Integrations available for: [Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents), [AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk), [Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk), [Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration), and [Keywords AI](https://docs.keywordsai.co/integration/development-frameworks/openai-agent).

## Core Concepts

Dive deeper into the key elements that power the OpenAI Agents SDK:

1.  **Agents**: LLMs configured with instructions, tools, guardrails, and handoffs
2.  **Handoffs**: A specialized tool call used by the Agents SDK for transferring control between agents
3.  **Guardrails**: Configurable safety checks for input and output validation
4.  **Sessions**: Automatic conversation history management across agent runs
5.  **Tracing**: Built-in tracking of agent runs, allowing you to view, debug and optimize your workflows

Explore the [examples](examples) directory to see the SDK in action, and read our [documentation](https://openai.github.io/openai-agents-python/) for more details.

## Getting Started

Follow these simple steps to set up your environment and begin using the OpenAI Agents SDK:

1.  **Set up your Python environment**:

    *   Option A: Using venv (traditional method)

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

    *   Option B: Using uv (recommended)

    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install the Agents SDK**:

    ```bash
    pip install openai-agents
    ```

    For voice support, install with the optional `voice` group: `pip install 'openai-agents[voice]'`.

## Example Usage

Quickly get up and running with these illustrative examples:

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

(_Ensure you set the `OPENAI_API_KEY` environment variable before running._)
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

## Agent Loop Explained

The `Runner.run()` function drives the agent workflow in a loop until a final output is generated:

1.  The LLM is called with the agent's settings and message history.
2.  The LLM returns a response that may include tool calls.
3.  If the response has a final output, it is returned, and the loop ends.
4.  If the response includes a handoff, control shifts to the new agent, and the process restarts from step 1.
5.  Tool calls are processed (if any), and tool responses are appended.  The loop then returns to step 1.

The `max_turns` parameter can be used to limit the number of loop iterations.

### Final Output

The final output represents the concluding step of the agent's operation:

1.  If an `output_type` is set on the agent, the final output is when the LLM returns a structured output that matches that type.
2.  If no `output_type` is set (i.e., plain text responses), the first LLM response that doesn't include tool calls or handoffs is considered the final output.

This means the agent loop functions as follows:

1.  If an agent has an `output_type`, the loop runs until the agent produces a structured output of that type.
2.  If an agent lacks an `output_type`, the loop continues until the agent delivers a message without any tool calls or handoffs.

## Agent Patterns

The Agents SDK is designed to support a wide array of LLM workflows including deterministic flows and iterative loops. See examples in [`examples/agent_patterns`](examples/agent_patterns).

## Tracing

The Agents SDK provides built-in tracing functionality to help track and debug your agent runs. Explore integrations with popular tracing tools: [Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents), [AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk), [Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk), [Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration), and [Keywords AI](https://docs.keywordsai.co/integration/development-frameworks/openai-agent). See [Tracing](http://openai.github.io/openai-agents-python/tracing) for detailed customization options and a broader list of [external tracing processors](http://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list).

## Long-Running Agents and Human-in-the-Loop

Integrate with [Temporal](https://temporal.io/) to create durable, long-running workflows, including human-in-the-loop tasks. Watch a demo video of Temporal and the Agents SDK in action [here](https://www.youtube.com/watch?v=fFBZqzT4DD8) and view the [docs here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).

## Sessions

The SDK includes session memory to automatically manage conversation history.

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

-   **No memory** (default): Omit the session parameter.
-   **`session: Session = DatabaseSession(...)`**: Use a Session instance to manage conversation history.

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

## Development (For contributing to the SDK)

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

We would like to acknowledge and give thanks to the open-source community, including:

-   [Pydantic](https://docs.pydantic.dev/latest/) (data validation) and [PydanticAI](https://ai.pydantic.dev/) (advanced agent framework)
-   [LiteLLM](https://github.com/BerriAI/litellm) (unified interface for 100+ LLMs)
-   [MkDocs](https://github.com/squidfunk/mkdocs-material)
-   [Griffe](https://github.com/mkdocstrings/griffe)
-   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

We are committed to continuously developing the Agents SDK as an open-source framework, to empower community members to build upon our approach.