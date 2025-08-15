# Build Advanced LLM Workflows with the OpenAI Agents SDK

**The OpenAI Agents SDK empowers developers to create sophisticated, multi-agent workflows with ease, offering a flexible and powerful framework for orchestrating LLMs.**  ([Original Repo](https://github.com/openai/openai-agents-python))

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

### Key Features:

*   **Agent-Driven Workflows:** Design agents with specific instructions, tools, guardrails, and handoffs to manage complex tasks.
*   **Provider Agnostic:** Supports OpenAI APIs, chat completions, and over 100+ LLMs, providing flexibility in model selection.
*   **Handoffs for Agent Collaboration:** Enable seamless control transfer between agents for sophisticated multi-agent interactions.
*   **Guardrails for Reliability:** Implement configurable safety checks for robust input and output validation, ensuring reliability and control.
*   **Session Management:** Utilize built-in session memory for persistent conversation history, streamlining multi-turn interactions.
*   **Built-in Tracing:**  Easily view, debug and optimize agent workflows with built-in tracing features and support for integrations with platforms like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI.
*   **Advanced Agent Patterns:** Model diverse workflows, including deterministic flows and iterative loops, to suit various LLM application needs.
*   **Long-Running Workflows and Human-in-the-Loop:** Run durable, long-running workflows with the integration with Temporal, including human-in-the-loop tasks.
*   **Customizable:** Extend the SDK with custom session memory implementations.

### Getting Started

1.  **Set up your Python environment:** Choose your preferred method (venv or uv).
    *   **Using venv:**

        ```bash
        python -m venv env
        source env/bin/activate  # On Windows: env\Scripts\activate
        ```

    *   **Using uv (Recommended):**

        ```bash
        uv venv
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        ```

2.  **Install the Agents SDK:**

    ```bash
    pip install openai-agents
    ```

    *   For voice support: `pip install 'openai-agents[voice]'`

### Example: "Hello World"

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

(*Ensure you set the `OPENAI_API_KEY` environment variable*)

### Example: Handoffs

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

### Example: Functions

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

### The Agent Loop

The `Runner.run()` function runs an agent loop until a final output is generated.

1.  An LLM is called, using the model and settings from the agent, along with the message history.
2.  The LLM returns a response, which may include tool calls.
3.  If the response has a final output (see below), it is returned and the loop ends.
4.  If the response has a handoff, the agent is set to the new agent and it goes back to step 1.
5.  If there are tool calls, those are processed and the tool responses are appended to the message history, then the loop restarts at step 1.

There is a `max_turns` parameter that you can use to limit the number of times the loop executes.

#### Final Output

1.  If an `output_type` is defined on the agent, the loop runs until the LLM returns that type.
2.  If no `output_type` is specified, the first LLM response without tool calls or handoffs is considered the final output.

### Common Agent Patterns

The SDK is designed to be flexible to support a wide range of LLM workflows. See the examples in [`examples/agent_patterns`](examples/agent_patterns).

### Tracing

The SDK automatically traces agent runs. Tracing can be extended by supporting custom spans and various external destinations, including [Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents), [AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk), [Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk), [Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration), and [Keywords AI](https://docs.keywordsai.co/integration/development-frameworks/openai-agent). For more details about how to customize or disable tracing, see [Tracing](http://openai.github.io/openai-agents-python/tracing), which also includes a larger list of [external tracing processors](http://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list).

### Long Running Agents & Human-in-the-Loop

You can use the Agents SDK [Temporal](https://temporal.io/) integration to run durable, long-running workflows, including human-in-the-loop tasks. View a demo of Temporal and the Agents SDK working in action to complete long-running tasks [in this video](https://www.youtube.com/watch?v=fFBZqzT4DD8), and [view docs here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).

### Sessions

The SDK provides built-in session memory to automatically maintain conversation history across multiple agent runs.

#### Quick Start

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

#### Session Options

*   **No memory** (default): No session memory when the session parameter is omitted.
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

#### Custom Session Implementations

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

### Development

(Only needed if you are editing the SDK/examples)

0.  Ensure [`uv`](https://docs.astral.sh/uv/) is installed.

    ```bash
    uv --version
    ```

1.  Install dependencies

    ```bash
    make sync
    ```

2.  Lint and test (after making changes):

    ```
    make check # run tests linter and typechecker
    ```

    or individually:

    ```
    make tests  # run tests
    make mypy   # run typechecker
    make lint   # run linter
    make format-check # run style checker
    ```

### Acknowledgements

Thank you to the open-source community, especially:

*   [Pydantic](https://docs.pydantic.dev/latest/) (data validation) and [PydanticAI](https://ai.pydantic.dev/) (advanced agent framework)
*   [LiteLLM](https://github.com/BerriAI/litellm) (unified interface for 100+ LLMs)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

We are committed to developing the Agents SDK as an open-source framework for the community to build upon.