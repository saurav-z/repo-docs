# OpenAI Agents SDK: Build Powerful Multi-Agent Workflows

**Unleash the power of multi-agent workflows with the OpenAI Agents SDK, a lightweight yet robust framework for orchestrating complex LLM interactions.**

[See the original repository](https://github.com/openai/openai-agents-python)

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

The OpenAI Agents SDK empowers developers to build sophisticated, multi-agent systems using various LLMs. This framework is designed to be provider-agnostic, offering support for the OpenAI Responses and Chat Completions APIs, and compatibility with over 100 other Large Language Models.

**Key Features:**

*   **Agents:** Configure LLMs with instructions, tools, guardrails, and handoffs to define agent behavior.
*   **Handoffs:** Seamlessly transfer control between agents for complex task orchestration.
*   **Guardrails:** Ensure safety and data integrity with configurable input and output validation.
*   **Sessions:** Built-in conversation history management for persistent and context-aware agent interactions.
*   **Tracing:**  Gain visibility into agent runs with built-in tracing for debugging, optimization, and integrations with tools like Logfire, AgentOps, Braintrust, Scorecard and Keywords AI.
*   **Tools:** Utilize function tools and integrate other external tools within the agent loop.
*   **Long Running Agents & Human-in-the-Loop:** Leverage Temporal integration to build durable, long-running workflows.

## Getting Started

Follow these steps to quickly set up and start using the Agents SDK:

1.  **Set up your Python Environment:** Choose between venv (traditional) or uv (recommended)
    *   **Using venv:**
        ```bash
        python -m venv env
        source env/bin/activate  # On Windows: env\Scripts\activate
        ```
    *   **Using uv:**
        ```bash
        uv venv
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        ```
2.  **Install the Agents SDK:**
    ```bash
    pip install openai-agents
    ```
    Install voice support: `pip install 'openai-agents[voice]'`

## Examples

Explore the power of the Agents SDK with these examples:

### "Hello World" Example

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

*(Ensure the `OPENAI_API_KEY` environment variable is set before running.)*

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

Explore additional examples in the [`examples`](examples) directory, and delve deeper into the details in the [documentation](https://openai.github.io/openai-agents-python/).

## Agent Loop Explained

The `Runner.run()` function executes a loop until a final output is reached:

1.  The LLM is called, using the model and settings on the agent, and the message history.
2.  The LLM returns a response, which may include tool calls.
3.  If the response has a final output, it's returned and the loop ends.
4.  If the response has a handoff, control is passed to the new agent and the loop restarts.
5.  Tool calls are processed, the tool responses are added, and the loop restarts.

The `max_turns` parameter can be used to limit the loop's executions.

### Final Output Explained

The final output is determined by the agent's design and configuration:

1.  If the agent has an `output_type`, the loop runs until output matching that type is produced.
2.  If there's no `output_type`, the loop runs until a message without tool calls or handoffs is produced.

## Common Agent Patterns

The Agents SDK offers flexibility to model various LLM workflows, including deterministic flows and iterative loops. Discover more in [`examples/agent_patterns`](examples/agent_patterns).

## Development (for SDK/example modification)

Follow these steps to contribute to the Agents SDK:

1.  Ensure [`uv`](https://docs.astral.sh/uv/) is installed.

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

    Or run them individually:

    ```
    make tests  # run tests
    make mypy   # run typechecker
    make lint   # run linter
    make format-check # run style checker
    ```

## Acknowledgements

The development team would like to thank the following open-source projects for their contributions:

*   [Pydantic](https://docs.pydantic.dev/latest/) and [PydanticAI](https://ai.pydantic.dev/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)