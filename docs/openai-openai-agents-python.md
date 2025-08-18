# OpenAI Agents SDK: Build Powerful Multi-Agent Workflows (Python)

**Empower your AI applications by building dynamic, multi-agent workflows with the flexible and versatile [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).**

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

This Python SDK offers a lightweight, provider-agnostic framework for creating complex AI-driven applications, seamlessly integrating with OpenAI's APIs, other leading LLMs, and various tools.

**Key Features:**

*   🤖 **Agents:** Define AI agents with specific instructions, tools, and guardrails.
*   🔄 **Handoffs:** Enable seamless control transfer between agents for complex task execution.
*   🛡️ **Guardrails:** Implement configurable safety checks for robust input and output validation.
*   💬 **Sessions:** Leverage automatic conversation history management for continuous dialogue.
*   📊 **Tracing:** Built-in tracing to view, debug, and optimize your agent workflows.
*   🔌 **Provider Agnostic:** Supports the OpenAI Responses and Chat Completions APIs and 100+ other LLMs.
*   💾 **Memory:** Built-in session memory that automatically maintains conversation history across multiple agent runs.

**Core Concepts:**

1.  [**Agents**](https://openai.github.io/openai-agents-python/agents): LLMs configured with instructions, tools, guardrails, and handoffs
2.  [**Handoffs**](https://openai.github.io/openai-agents-python/handoffs/): A specialized tool call used by the Agents SDK for transferring control between agents
3.  [**Guardrails**](https://openai.github.io/openai-agents-python/guardrails/): Configurable safety checks for input and output validation
4.  [**Sessions**](#sessions): Automatic conversation history management across agent runs
5.  [**Tracing**](https://openai.github.io/openai-agents-python/tracing/): Built-in tracking of agent runs, allowing you to view, debug and optimize your workflows

**Getting Started:**

1.  **Set up your Python environment:**

    *   **Option A: venv**
        ```bash
        python -m venv env
        source env/bin/activate  # On Windows: env\Scripts\activate
        ```
    *   **Option B: uv (recommended)**
        ```bash
        uv venv
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        ```

2.  **Install the Agents SDK:**

    ```bash
    pip install openai-agents
    ```

    *For voice support:* `pip install 'openai-agents[voice]'`

**Example: Hello World**

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
```

*(Ensure you set the `OPENAI_API_KEY` environment variable).*

**Explore more examples in the [examples](examples) directory.**

**Advanced Features:**

*   **Handoffs:**
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
*   **Functions:**
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

*   **Tracing:** Monitor and debug agent behavior, with integrations for Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI.  For more details about how to customize or disable tracing, see [Tracing](http://openai.github.io/openai-agents-python/tracing), which also includes a larger list of [external tracing processors](http://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list).

*   **Long Running Agents & Human-in-the-Loop:** Integrates with [Temporal](https://temporal.io/) for durable, long-running workflows. View a demo of Temporal and the Agents SDK working in action to complete long-running tasks [in this video](https://www.youtube.com/watch?v=fFBZqzT4DD8), and [view docs here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).

*   **Sessions:**

    *   **Quick Start:**

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
    *   **Session options:**
        -   **No memory** (default): No session memory when session parameter is omitted
        -   **`session: Session = DatabaseSession(...)`**: Use a Session instance to manage conversation history

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

    *   **Custom session implementations:**

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

**Development** (for contributing to the SDK):

1.  Ensure you have `uv` installed:  `uv --version`
2.  Install dependencies: `make sync`
3.  Run tests, lint, and type checks:  `make check` (or individual commands like `make tests`, `make mypy`, etc.)

**Acknowledgements:**

We gratefully acknowledge the contributions of the open-source community, especially: Pydantic, PydanticAI, LiteLLM, MkDocs, Griffe, uv, and ruff.

Contribute to the development of the OpenAI Agents SDK!