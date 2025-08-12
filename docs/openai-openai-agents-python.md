# Build Advanced LLM Workflows with the OpenAI Agents SDK

**Empower your applications with sophisticated multi-agent orchestration using the OpenAI Agents SDK, a lightweight and versatile framework for building intelligent workflows.**  [Explore the OpenAI Agents SDK on GitHub](https://github.com/openai/openai-agents-python).

![OpenAI Agents SDK Overview](https://cdn.openai.com/API/docs/images/orchestration.png)

The OpenAI Agents SDK provides a robust and adaptable framework for constructing multi-agent systems. This SDK seamlessly integrates with OpenAI's APIs and supports over 100 LLMs, allowing you to create intricate workflows with ease.

**Key Features:**

*   **Agents:** Configure LLMs with instructions, tools, guardrails, and handoffs to create specialized agents.
*   **Handoffs:** Seamlessly transfer control between agents to create dynamic and adaptive workflows.
*   **Guardrails:** Implement customizable safety checks for input and output validation, ensuring responsible AI interactions.
*   **Sessions:** Maintain automatic conversation history across agent runs for context-aware interactions.
*   **Tracing:** Built-in tracing capabilities for comprehensive workflow monitoring, debugging, and optimization, with integrations for tools like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI.

## Get Started

**1. Set up your Python environment:**

*   **Option A: Using venv (traditional):**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```
*   **Option B: Using uv (recommended):**

    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

**2. Install the Agents SDK:**

```bash
pip install openai-agents
```

For voice support, install with the optional `voice` group: `pip install 'openai-agents[voice]'`.

## Quick Examples

**Hello World:**

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

**Handoffs Example:**

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

**Functions Example:**

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

## Agent Loop Logic

The `Runner.run()` function orchestrates the agent loop:

1.  The LLM is called using the agent's model and settings.
2.  The LLM response is processed, potentially including tool calls.
3.  If the response has a final output, it is returned, and the loop ends.
4.  If a handoff is present, control is transferred to the new agent, and the loop restarts.
5.  Tool calls are executed, and responses are added.

The `max_turns` parameter can be used to limit the loop's execution.

### Final Output Definition

The final output is the result of the agent loop:

1.  If an `output_type` is set, the loop continues until a structured output of that type is produced.
2.  If there's no `output_type`, the loop ends when an LLM response without tool calls or handoffs is generated.

## Common Agent Patterns

The Agents SDK supports various LLM workflow patterns, including deterministic flows and iterative loops.  See more in the `examples/agent_patterns` directory.

## Tracing and Monitoring

The SDK includes built-in tracing to track agent behavior.  Customize tracing with external destinations such as:

*   [Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents)
*   [AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk)
*   [Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk)
*   [Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration)
*   [Keywords AI](https://docs.keywordsai.co/integration/development-frameworks/openai-agent)

For details, see the [Tracing](http://openai.github.io/openai-agents-python/tracing/) documentation.

## Long-Running Agents & Human-in-the-Loop

Integrate with [Temporal](https://temporal.io/) for durable, long-running workflows and human-in-the-loop tasks.  See the [video](https://www.youtube.com/watch?v=fFBZqzT4DD8) demo and [docs](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents) for more.

## Sessions

The Agents SDK offers built-in session memory:

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

*   **No memory** (default):  Omit the session parameter.
*   **`session: Session = DatabaseSession(...)`**: Use a Session instance.

### Custom Session Implementations

Create custom session memory by implementing the `Session` protocol.

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

1.  Ensure `uv` is installed.
2.  Install dependencies: `make sync`
3.  Lint/test: `make check` (or individual commands: `make tests`, `make mypy`, `make lint`, `make format-check`)

## Acknowledgements

The project recognizes the contributions of the open-source community, particularly:

*   [Pydantic](https://docs.pydantic.dev/latest/) & [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)