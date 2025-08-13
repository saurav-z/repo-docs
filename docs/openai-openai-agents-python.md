# OpenAI Agents SDK: Build Powerful Multi-Agent Workflows with Ease

**Unleash the power of collaborative AI with the OpenAI Agents SDK, a flexible and provider-agnostic framework for constructing advanced multi-agent systems.** ([Original Repo](https://github.com/openai/openai-agents-python))

[<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">](https://github.com/openai/openai-agents-python)

**Key Features:**

*   **Agent-Based Architecture:** Define and orchestrate LLMs with instructions, tools, and handoffs for complex tasks.
*   **Handoffs:** Seamlessly transfer control between agents to create dynamic workflows.
*   **Guardrails:** Implement configurable safety checks for input and output validation, ensuring responsible AI usage.
*   **Sessions:** Leverage automatic conversation history management for persistent and context-aware agent interactions.
*   **Tracing:**  Built-in tracing for comprehensive monitoring, debugging, and optimization of agent runs, with integrations for popular tools like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI.
*   **Temporal Integration:** Run durable, long-running workflows with human-in-the-loop tasks using [Temporal](https://temporal.io/).
*   **Flexible:** Supports OpenAI APIs, Chat Completions APIs, and 100+ other LLMs.

**Core Concepts:**

*   **Agents:** LLMs configured with instructions, tools, guardrails, and handoffs
*   **Handoffs:**  A specialized tool call used by the Agents SDK for transferring control between agents
*   **Guardrails:** Configurable safety checks for input and output validation
*   **Sessions:** Automatic conversation history management across agent runs
*   **Tracing:** Built-in tracking of agent runs, allowing you to view, debug and optimize your workflows

**Get Started Quickly:**

1.  **Set up your Python environment:**

    *   **Using `venv` (traditional):**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

    *   **Using `uv` (recommended):**

    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install the Agents SDK:**

    ```bash
    pip install openai-agents
    ```

    *   For voice support: `pip install 'openai-agents[voice]'`

**Example: "Hello World" Agent**

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
```
(_Ensure you set the `OPENAI_API_KEY` environment variable_)

**Explore More:**

*   [Examples Directory](examples): Discover practical examples showcasing the SDK's capabilities.
*   [Documentation](https://openai.github.io/openai-agents-python/): Deep dive into the SDK's features and functionalities.

**Development (for contributing or SDK modification):**

0.  Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.

    ```bash
    uv --version
    ```

1.  Install dependencies

    ```bash
    make sync
    ```

2.  Lint/test (after changes)

    ```
    make check # run tests linter and typechecker
    ```

    Or run individually:

    ```
    make tests  # run tests
    make mypy   # run typechecker
    make lint   # run linter
    make format-check # run style checker
    ```

**Acknowledgements:**

The OpenAI Agents SDK benefits from the contributions of the open-source community, including:

*   [Pydantic](https://docs.pydantic.dev/latest/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)