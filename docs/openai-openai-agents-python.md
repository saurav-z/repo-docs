# OpenAI Agents SDK: Build Powerful Multi-Agent Workflows with Python

**Unlock the potential of multi-agent systems with the OpenAI Agents SDK, a flexible framework for building, orchestrating, and optimizing LLM-powered workflows.** ([Original Repo](https://github.com/openai/openai-agents-python))

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

> **Looking for the JavaScript/TypeScript version? Check out [Agents SDK JS/TS](https://github.com/openai/openai-agents-js).**

## Key Features

*   **Provider-Agnostic:** Works seamlessly with OpenAI Responses and Chat Completions APIs, as well as 100+ other LLMs.
*   **Agents:** Define and configure LLMs with instructions, tools, guardrails, and handoffs for complex tasks.
*   **Handoffs:** Facilitate control transfer between agents, enabling dynamic workflow management.
*   **Guardrails:** Implement configurable safety checks for input and output validation, ensuring reliable and secure interactions.
*   **Sessions:** Maintain automatic conversation history across agent runs for contextual awareness and smoother interactions.
*   **Tracing:** Built-in tracing for easy debugging, optimization, and visualization of agent workflows, with integrations for leading observability platforms.
*   **Long-Running Workflows:** Integrate with Temporal for durable, human-in-the-loop tasks and long-running processes.

## Core Concepts

1.  [**Agents**](https://openai.github.io/openai-agents-python/agents): Configured LLMs with instructions, tools, guardrails, and handoffs.
2.  [**Handoffs**](https://openai.github.io/openai-agents-python/handoffs/): A tool call for transferring control between agents.
3.  [**Guardrails**](https://openai.github.io/openai-agents-python/guardrails/): Configurable safety checks for input and output validation.
4.  **Sessions:** Automatic conversation history management across agent runs.
5.  [**Tracing**](https://openai.github.io/openai-agents-python/tracing/): Built-in tracking of agent runs for viewing, debugging, and optimizing workflows.

## Get Started

### 1. Set up your Python environment

Choose your preferred method:

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

### 2. Install Agents SDK

```bash
pip install openai-agents
```

For voice support, install with the optional `voice` group: `pip install 'openai-agents[voice]'`.

## Examples

Explore these examples to quickly learn and experiment with the OpenAI Agents SDK:

### Hello World

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
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


if __name__ == "__main__":
    asyncio.run(main())
```

## Agent Loop & Final Output

The `Runner.run()` function orchestrates the agent loop:

1.  Call the LLM with agent settings and message history.
2.  LLM returns a response, which may include tool calls.
3.  If the response has a final output, it is returned and the loop ends.
4.  If the response has a handoff, the agent switches and the loop returns to step 1.
5.  Process tool calls and responses, then go back to step 1.

The `max_turns` parameter limits the number of loop iterations.

*   **Final Output:**
    1.  If the agent has an `output_type`, the loop runs until structured output of that type is returned.
    2.  If there is no `output_type`, the loop runs until a message without tool calls or handoffs is generated.

## Common Agent Patterns

The SDK supports various LLM workflows including deterministic flows and iterative loops. Explore example patterns in [`examples/agent_patterns`](examples/agent_patterns).

## Tracing

Automatically trace and debug your agent runs. Integrate with various external destinations, including [Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents), [AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk), [Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk), [Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration), and [Keywords AI](https://docs.keywordsai.co/integration/development-frameworks/openai-agent). See [Tracing](http://openai.github.io/openai-agents-python/tracing) and [external tracing processors](http://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list) for customization.

## Long Running Agents & Human-in-the-Loop

Utilize the [Temporal](https://temporal.io/) integration for durable workflows, including human-in-the-loop tasks. See a demo [in this video](https://www.youtube.com/watch?v=fFBZqzT4DD8) and [view docs here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).

## Sessions

Maintain context across runs.

*   **No memory** (default): Omit the session parameter.
*   **`session: Session = DatabaseSession(...)`**: Use a Session instance to manage history.

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
-   **`session: Session = DatabaseSession(...)`**: Use a Session instance to manage history.

### Custom Session Implementations

Implement your own session memory by creating a class following the `Session` protocol.

## Development

(Only needed for SDK/example modifications)

1.  Ensure [`uv`](https://docs.astral.sh/uv/) is installed.
2.  Install dependencies: `make sync`.
3.  Run linting, testing, and type checking: `make check`.
    *   Individually: `make tests`, `make mypy`, `make lint`, `make format-check`

## Acknowledgements

This project acknowledges the contributions of the open-source community, especially:

*   [Pydantic](https://docs.pydantic.dev/latest/) and [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)

The project is committed to continuing to be an open-source framework.