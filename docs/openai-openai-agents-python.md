# Build Advanced AI Workflows with the OpenAI Agents SDK

**[Explore the OpenAI Agents SDK on GitHub](https://github.com/openai/openai-agents-python) and unlock the power of multi-agent systems to create dynamic and intelligent applications.**

The OpenAI Agents SDK is a robust Python framework designed to simplify the creation of multi-agent workflows. It provides a flexible and extensible architecture, supporting a wide variety of language models and powerful features.

**Key Features:**

*   **Agent-Based Architecture:** Define agents with specific instructions, tools, and guardrails to orchestrate complex tasks.
*   **Handoffs:** Seamlessly transfer control between agents to facilitate collaborative problem-solving.
*   **Guardrails:** Implement safety checks and validation to ensure reliable and secure outputs.
*   **Sessions:** Manage conversation history automatically across agent runs for context-aware interactions.
*   **Tracing:** Monitor, debug, and optimize your agent workflows with built-in tracing capabilities and integrations with popular tools.
*   **Flexible LLM Support:** Compatible with OpenAI's APIs and 100+ other LLMs via LiteLLM.
*   **Long-Running Agents & Human-in-the-Loop:** Integrates with Temporal for durable workflows, including human-in-the-loop tasks.
*   **Built-in Session Memory:** Automatically maintains conversation history, eliminating the need to manually manage `.to_input_list()` between turns.
*   **Custom Session Implementations:** Create your own session memory by creating a class that follows the `Session` protocol.

**Getting Started:**

1.  **Set up your Python environment:** Use either `venv` or `uv` (recommended) to create and activate a virtual environment.
2.  **Install the SDK:**

```bash
pip install openai-agents
```

   For voice support:
```bash
pip install 'openai-agents[voice]'
```

**Example: Hello World**

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
```

**Additional Resources:**

*   **Documentation:** Dive deeper into the SDK's features and usage with the official [documentation](https://openai.github.io/openai-agents-python/).
*   **Examples:** Explore the `examples` directory for practical demonstrations of the SDK in action.