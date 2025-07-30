# Build Powerful Multi-Agent Workflows with the OpenAI Agents SDK

**Unlock the potential of multi-agent systems with the OpenAI Agents SDK, a flexible and powerful Python framework.** ([View on GitHub](https://github.com/openai/openai-agents-python))

The OpenAI Agents SDK is your key to building sophisticated multi-agent workflows with ease. It's designed to be provider-agnostic, supporting the OpenAI Responses and Chat Completions APIs, alongside over 100 other Large Language Models (LLMs).

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

**Key Features:**

*   **Agents:** Configure LLMs with instructions, tools, and guardrails to accomplish specific tasks.
*   **Handoffs:** Seamlessly transfer control between agents for complex workflows.
*   **Guardrails:** Implement configurable safety checks for input and output validation, ensuring reliable and secure agent interactions.
*   **Sessions:** Manage conversation history automatically across agent runs, allowing for continuity and context retention.
*   **Tracing:** Built-in tracing capabilities allow you to easily view, debug, and optimize your agent workflows. Compatible with Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI for comprehensive monitoring and analysis.

**Core Concepts:**

1.  [**Agents**](https://openai.github.io/openai-agents-python/agents): The building blocks of your workflows, with defined instructions, tools, and handoffs.
2.  [**Handoffs**](https://openai.github.io/openai-agents-python/handoffs/): Enables the transfer of control between agents, streamlining complex tasks.
3.  [**Guardrails**](https://openai.github.io/openai-agents-python/guardrails/): Safety checks for input and output validation.
4.  [**Sessions**](#sessions): Provides automatic conversation history management.
5.  [**Tracing**](https://openai.github.io/openai-agents-python/tracing/): Built-in tracking of agent runs.

**Get Started:**

1.  **Set up your Python environment:**

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

2.  **Install the Agents SDK:**

    ```bash
    pip install openai-agents
    ```

    For voice support, install with the optional `voice` group: `pip install 'openai-agents[voice]'`.

**Hello World Example:**

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

**The Agent Loop:**

The `Runner.run()` method orchestrates the execution of your agents, managing the loop until a final output is achieved.

1.  The LLM is called with the model, agent settings, and message history.
2.  The LLM's response, which can include tool calls, is returned.
3.  If the response yields a final output (based on `output_type`), the output is returned, ending the loop.
4.  If a handoff is included, the current agent transitions to the new agent, and the process restarts.
5.  Tool calls are processed (if any), and tool responses are appended to the messages before the loop begins again.

The `max_turns` parameter can be used to limit the loop iterations.

**Final Output:**

*   If the agent has an `output_type`, the loop runs until the LLM returns a structured output of that type (using structured outputs).
*   If the agent lacks an `output_type`, the loop continues until the LLM response doesn't contain any tool calls or handoffs.

**Common Agent Patterns:**

The Agents SDK provides flexibility in modeling various LLM workflows, including deterministic flows and iterative loops. Explore example patterns in [`examples/agent_patterns`](examples/agent_patterns).

**Tracing:**

The SDK includes an auto-tracing feature to monitor and debug your agent behaviors.

**Sessions:**

Built-in session memory automatically retains conversation history for subsequent agent runs, eliminating the need for manual management between turns.

**Quick Start:**

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

**Session Options:**

*   **No memory** (default): No session memory when the session parameter is omitted.
*   **`session: Session = DatabaseSession(...)`**: Utilize a Session instance to manage conversation history.

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

**Custom Session Implementations:**
Develop custom session memory by creating a class based on the `Session` protocol.

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

**Development** (For SDK/example modification only):

0.  Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.
1.  Install dependencies: `make sync`
2.  (After changes) lint/test: `make check` (runs tests, linter, and typechecker) or run them individually.

**Acknowledgements:**

We acknowledge the contributions of the open-source community, especially:

*   [Pydantic](https://docs.pydantic.dev/latest/) and [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

We are committed to enhancing the Agents SDK as an open-source project, welcoming community participation.