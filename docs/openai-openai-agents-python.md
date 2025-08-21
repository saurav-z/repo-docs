# Build Powerful Multi-Agent Workflows with the OpenAI Agents SDK

**Unlock the power of AI with the OpenAI Agents SDK, a lightweight, provider-agnostic framework for orchestrating complex multi-agent workflows.**  [Explore the original repository on GitHub](https://github.com/openai/openai-agents-python).

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

**Key Features:**

*   **Agent Definition:** Configure LLMs with specific instructions, tools, and guardrails for focused tasks.
*   **Handoffs:** Seamlessly transfer control between agents to create dynamic workflows.
*   **Guardrails:** Ensure safety and data integrity with configurable input/output validation.
*   **Sessions:** Maintain context and conversation history across multiple agent runs with built-in session memory.
*   **Tracing:** Built-in tracing for in-depth monitoring, debugging, and optimization of agent behavior. Supports integrations with popular platforms like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI.
*   **Extensible Architecture:** Adapt and expand workflows with flexibility.
*   **Built-in support for over 100+ LLMs and APIs:**  Using the OpenAI Responses and Chat Completions APIs

### Get Started

1.  **Set up your Python environment:** Requires Python 3.9 or newer.
2.  **Install the SDK:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install openai-agents
    ```
    *Optional voice support:* `pip install 'openai-agents[voice]'`

    *Alternatively, using uv:*

    ```bash
    uv init
    uv add openai-agents
    ```
    *Optional voice support:* `uv add 'openai-agents[voice]'`

### Hello World Example

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

### Advanced Features:

*   **Handoffs Example:** Demonstrate agent-to-agent communication.
*   **Functions Example:** Integrate external tools and data into your agents.
*   **Agent Loop:** Understand the core logic of agent execution.
*   **Common Agent Patterns:** Explore advanced workflow design using example code.
*   **Tracing:** Comprehensive tracing for monitoring and debugging your agents.
*   **Long-Running Agents & Human-in-the-Loop:**  Integrate with [Temporal](https://temporal.io/) for durable workflows.
*   **Sessions:**  Effortlessly manage conversation history.  Choose from no memory, SQLite, or implement your custom session.

### Development

Instructions are provided for those looking to contribute to the SDK, including environment setup and testing procedures.

### Acknowledgements

This project benefits from the contributions of the open-source community.

*   [Pydantic](https://docs.pydantic.dev/latest/)
*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [MkDocs](https://github.com/squidfunk/mkdocs-material)
*   [Griffe](https://github.com/mkdocstrings/griffe)
*   [uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff)