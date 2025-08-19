# Build Powerful AI Workflows with the OpenAI Agents SDK

**Effortlessly create and manage multi-agent workflows using the OpenAI Agents SDK, a flexible and provider-agnostic Python framework.** Learn more about the original repo [here](https://github.com/openai/openai-agents-python).

**Key Features:**

*   **Agent Definition:** Easily configure LLMs with instructions, tools, and guardrails to guide their behavior.
*   **Handoffs:** Seamlessly transfer control between agents for complex task execution.
*   **Guardrails:** Implement configurable safety checks for robust input and output validation.
*   **Sessions:** Automatically manage conversation history for continuous and context-aware agent interactions.
*   **Tracing:** Built-in tracing enables comprehensive monitoring, debugging, and optimization of your agent workflows.
*   **Provider-Agnostic:** Compatible with OpenAI APIs, Chat Completions APIs, and 100+ other LLMs.
*   **Long Running Agents & Human-in-the-Loop**: Supports durable long-running workflows, including human-in-the-loop tasks using a Temporal integration.

## Quickstart

**Install the SDK:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install openai-agents
```

Or using `uv`:

```bash
uv init
uv add openai-agents
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

## Core Concepts

*   **Agents**: Configure LLMs with instructions, tools, and guardrails for specific tasks.
*   **Handoffs**: Delegate tasks between agents.
*   **Guardrails**: Ensure data quality and safety.
*   **Sessions**: Maintain conversational context across agent runs.
*   **Tracing**: Track agent behavior for debugging and optimization.

##  Explore Agent Patterns

The Agents SDK is designed for flexibility, allowing you to model various LLM workflows:

*   **Deterministic Flows:** Implement predictable agent interactions.
*   **Iterative Loops:** Create agents that can perform repeated actions.

Check the `examples/agent_patterns` directory for practical use cases.

## Advanced Features

*   **Tracing Integration:** Integrate with tracing tools like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI for detailed workflow analysis. (See: [Tracing](http://openai.github.io/openai-agents-python/tracing/))
*   **Temporal Integration**: Run durable, long-running workflows, including human-in-the-loop tasks using a Temporal integration. View a demo of Temporal and the Agents SDK [in this video](https://www.youtube.com/watch?v=fFBZqzT4DD8) and [view docs here](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents).
*   **Session Management**: Automatically maintain conversation history. This allows the user to provide context and allows the agent to build upon previous interactions.

## Development

Follow these steps to contribute or modify the SDK:

1.  Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.
2.  Install Dependencies: `make sync`
3.  (After Changes) Validate with: `make check`

Run individual checks with:

*   `make tests` (run tests)
*   `make mypy` (run typechecker)
*   `make lint` (run linter)
*   `make format-check` (run style checker)

## Acknowledgements

The OpenAI Agents SDK leverages the following open-source projects:

*   Pydantic and PydanticAI
*   LiteLLM
*   MkDocs
*   Griffe
*   uv and ruff