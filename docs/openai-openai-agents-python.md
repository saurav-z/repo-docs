# Build Powerful Multi-Agent Workflows with the OpenAI Agents SDK

**The OpenAI Agents SDK provides a flexible and provider-agnostic framework for building complex multi-agent workflows using Python, enabling you to orchestrate and automate tasks with ease.** (See the original repo: [OpenAI Agents SDK](https://github.com/openai/openai-agents-python))

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

Key Features:

*   **Agent-Centric Design:** Configure LLMs (including OpenAI and 100+ others) with instructions, tools, guardrails, and handoffs to build sophisticated AI assistants.
*   **Seamless Handoffs:** Easily transfer control between agents using specialized tool calls for dynamic workflow management.
*   **Robust Guardrails:** Implement configurable safety checks for input and output validation, ensuring responsible AI interactions.
*   **Intelligent Sessions:** Leverage automatic conversation history management across agent runs for contextual awareness and improved performance.
*   **Advanced Tracing:** Built-in tracing capabilities provide detailed insights into agent behavior, enabling debugging, optimization, and integration with various tracing platforms (Logfire, AgentOps, Braintrust, Scorecard, Keywords AI, and custom integrations).

## Core Concepts

1.  [**Agents**](https://openai.github.io/openai-agents-python/agents): LLMs configured with instructions, tools, guardrails, and handoffs
2.  [**Handoffs**](https://openai.github.io/openai-agents-python/handoffs/): A specialized tool call used by the Agents SDK for transferring control between agents
3.  [**Guardrails**](https://openai.github.io/openai-agents-python/guardrails/): Configurable safety checks for input and output validation
4.  [**Sessions**](#sessions): Automatic conversation history management across agent runs
5.  [**Tracing**](https://openai.github.io/openai-agents-python/tracing/): Built-in tracking of agent runs, allowing you to view, debug and optimize your workflows

## Get Started

1.  **Set up your Python environment:**

    *   **Option A: Using `venv` (traditional):**

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

    *   For voice support, install with the optional `voice` group: `pip install 'openai-agents[voice]'`.

## Examples

Explore the [examples](examples) directory to see the SDK in action, and read our [documentation](https://openai.github.io/openai-agents-python/) for more details.

*   **Hello World:**

    ```python
    from agents import Agent, Runner

    agent = Agent(name="Assistant", instructions="You are a helpful assistant")

    result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)
    ```

    (_Ensure the `OPENAI_API_KEY` environment variable is set._)

*   **Handoffs Example:** (Shows how to transfer control between agents based on input language)

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

*   **Functions Example:** (Demonstrates how to use tools)

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

The `Runner.run()` function orchestrates the agent's execution loop until a final output is generated, integrating LLM calls, tool calls, and handoffs.

### Final Output

The final output is the final result of the agent's operation.

1.  If you set an `output_type` on the agent, the final output is when the LLM returns something of that type. We use [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) for this.
2.  If there's no `output_type` (i.e. plain text responses), then the first LLM response without any tool calls or handoffs is considered as the final output.

## Common Agent Patterns

The SDK supports various LLM workflow patterns including deterministic flows, iterative loops. See examples in [`examples/agent_patterns`](examples/agent_patterns).

## Long Running Agents & Human-in-the-Loop

The OpenAI Agents SDK can be integrated with [Temporal](https://temporal.io/) to run durable, long-running workflows, including human-in-the-loop tasks. Watch [this video](https://www.youtube.com/watch?v=fFBZqzT4DD8) demo and read the documentation [here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).

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

You can create your own session memory by implementing the `Session` protocol:

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

## Development (for contributing to the SDK)

1.  Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.
2.  Install dependencies

    ```bash
    make sync
    ```
3.  Lint, test, and typecheck after making changes:

    ```bash
    make check
    ```

## Acknowledgements

The OpenAI Agents SDK leverages and acknowledges the contributions of the open-source community, especially:

*   [Pydantic](https://docs.pydantic.dev/latest/) and [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)