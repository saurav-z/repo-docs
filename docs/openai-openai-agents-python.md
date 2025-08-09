# Build Powerful Multi-Agent Workflows with the OpenAI Agents SDK

**Unleash the power of multi-agent workflows with the OpenAI Agents SDK, a versatile Python framework for orchestrating intelligent conversations and automating complex tasks.** This framework, available on [GitHub](https://github.com/openai/openai-agents-python), allows developers to create agent-based systems.

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

**Key Features:**

*   **Agent-Driven Design:** Build sophisticated workflows by defining individual agents with specific instructions, tools, and guardrails.
*   **Provider Agnostic:** Works seamlessly with the OpenAI Responses and Chat Completions APIs, as well as 100+ other LLMs, offering flexibility in your choice of language models.
*   **Handoffs:**  Facilitate dynamic agent control by enabling agents to seamlessly transfer tasks and responsibilities to other agents, streamlining complex processes.
*   **Guardrails:** Implement configurable safety checks for input and output validation, ensuring the reliability and safety of your agent interactions.
*   **Sessions:**  Utilize automatic conversation history management across agent runs for seamless, context-aware interactions.
*   **Tracing:** Monitor and debug agent runs with built-in tracing capabilities, including integrations with popular tools like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI.

**Core Concepts:**

1.  [**Agents**](https://openai.github.io/openai-agents-python/agents): LLMs configured with instructions, tools, guardrails, and handoffs
2.  [**Handoffs**](https://openai.github.io/openai-agents-python/handoffs/): A specialized tool call used by the Agents SDK for transferring control between agents
3.  [**Guardrails**](https://openai.github.io/openai-agents-python/guardrails/): Configurable safety checks for input and output validation
4.  **Sessions**: Automatic conversation history management across agent runs
5.  [**Tracing**](https://openai.github.io/openai-agents-python/tracing/): Built-in tracking of agent runs, allowing you to view, debug and optimize your workflows

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

    *   For voice support: `pip install 'openai-agents[voice]'`

**Explore the [examples](examples) directory to see the SDK in action and dive deeper into the documentation at [OpenAI Agents Python Documentation](https://openai.github.io/openai-agents-python/) for more details.**

**(Remaining content of the original README, including "Hello world example", "Handoffs example", etc., is included to provide full context)**