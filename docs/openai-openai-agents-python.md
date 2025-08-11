# Build Powerful Multi-Agent Workflows with the OpenAI Agents SDK

**The OpenAI Agents SDK empowers developers to effortlessly create and orchestrate multi-agent workflows, offering a flexible and provider-agnostic solution for building sophisticated LLM-powered applications.  Explore the original repo [here](https://github.com/openai/openai-agents-python)!**

The OpenAI Agents SDK provides a lightweight yet feature-rich framework for building multi-agent systems, offering a provider-agnostic approach that supports OpenAI APIs, and 100+ other LLMs.

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

**Key Features:**

*   **Agent Definition:** Configure agents with instructions, tools, guardrails, and handoffs for dynamic behavior.
*   **Handoffs:** Seamlessly transfer control between agents for complex task execution.
*   **Guardrails:** Implement configurable safety checks for robust input and output validation.
*   **Sessions:** Leverage automatic conversation history management across agent runs for context-aware interactions.
*   **Tracing:** Gain insights into agent runs with built-in tracing for debugging, optimization, and external integrations.
*   **Flexible Integration:** Integrates with popular tools like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI for expanded functionality.
*   **Long-Running Workflows:** Supports long-running agent workflows and human-in-the-loop tasks with Temporal integration.

## Core Concepts

*   **Agents:** LLMs configured with instructions, tools, guardrails, and handoffs
*   **Handoffs:** A specialized tool call used by the Agents SDK for transferring control between agents
*   **Guardrails:** Configurable safety checks for input and output validation
*   **Sessions:** Automatic conversation history management across agent runs
*   **Tracing:** Built-in tracking of agent runs, allowing you to view, debug and optimize your workflows

## Getting Started

**Prerequisites:** Python 3.8 or higher

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

## Examples

*   **Hello World**

    ```python
    from agents import Agent, Runner

    agent = Agent(name="Assistant", instructions="You are a helpful assistant")

    result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)

    # Code within the code,
    # Functions calling themselves,
    # Infinite loop's dance.
    ```

    (_Ensure you set the `OPENAI_API_KEY` environment variable before running this.)_

    (_For Jupyter notebook users, see [hello_world_jupyter.ipynb](examples/basic/hello_world_jupyter.ipynb)_)

*   **Handoffs Example**

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

*   **Functions Example**

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

The `Runner.run()` method initiates the agent loop, which continues until a final output is generated.

1.  The LLM is called using the model and settings defined in the agent, along with the message history.
2.  The LLM response is processed, which might include tool calls.
3.  If the response contains a final output, it's returned, and the loop ends.
4.  If the response includes a handoff, the agent is set to the new agent, and the loop restarts from step 1.
5.  Tool calls are processed (if any), and tool responses are appended to the messages. Then, the loop returns to step 1.

The `max_turns` parameter allows you to limit the number of loop executions.

### Final Output

Final output is the last result produced by the agent within the loop:

1.  If the agent has an `output_type` defined, the loop runs until the agent produces a structured output that matches that type.
2.  If there's no `output_type` (plain text responses), the first LLM response without tool calls or handoffs is the final output.

## Agent Patterns

The Agents SDK supports versatile LLM workflows, including deterministic flows, iterative loops, and more. Explore the `examples/agent_patterns` directory for examples.

## Tracing Integration

The SDK features automatic tracing for monitoring and debugging agent behavior. Customize tracing with custom spans and integrations with tools like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI. See [Tracing](http://openai.github.io/openai-agents-python/tracing) for details.

## Long-Running Agents & Human-in-the-Loop

Integrate with [Temporal](https://temporal.io/) for durable, long-running workflows, including human-in-the-loop tasks. View the demo video [here](https://www.youtube.com/watch?v=fFBZqzT4DD8) and documentation [here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).

## Sessions

The SDK uses session memory to maintain conversation history across agent runs:

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

### Custom Session Implementations

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

## Development

(For editing the SDK/examples)

1.  Ensure [`uv`](https://docs.astral.sh/uv/) is installed.
2.  Install dependencies:

    ```bash
    make sync
    ```

3.  (After changes) Run linters and tests:

    ```bash
    make check
    ```

    Or run them individually:

    ```bash
    make tests  # run tests
    make mypy   # run typechecker
    make lint   # run linter
    make format-check # run style checker
    ```

## Acknowledgements

The SDK builds upon the excellent work of the open-source community, particularly:

*   [Pydantic](https://docs.pydantic.dev/latest/) and [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)