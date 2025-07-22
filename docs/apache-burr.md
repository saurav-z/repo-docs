# Burr: Build Stateful AI Applications with Ease

**Burr is a Python library that simplifies the development of stateful AI applications, offering a robust framework for managing state, monitoring execution, and integrating with your favorite tools.**

[Explore the Burr Repository](https://github.com/apache/burr)

<div>
  <a href="https://discord.gg/6Zy2DwP4f3" target="_blank">
    <img src="https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord" alt="Join Burr Discord">
  </a>
  <a href="https://pepy.tech/project/burr" target="_blank">
    <img src="https://static.pepy.tech/badge/burr/month" alt="PyPI Downloads">
  </a>
  <img src="https://static.pepy.tech/badge/burr" alt="PyPI Downloads">
  <a href="https://github.com/dagworks-inc/burr/pulse" target="_blank">
    <img src="https://img.shields.io/github/last-commit/dagworks-inc/burr" alt="GitHub Last Commit">
  </a>
  <a href="https://twitter.com/burr_framework" target="_blank">
    <img src="https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social" alt="Follow @burr_framework on X">
  </a>
  <a href="https://www.linkedin.com/showcase/dagworks-inc" target="_blank">
    <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin" alt="Follow DAGWorks on LinkedIn">
  </a>
  <a href="https://twitter.com/burr_framework" target="_blank">
    <img src="https://img.shields.io/badge/burr_framework-Follow-purple.svg?logo=X" alt="Follow @burr_framework on X">
  </a>
  <a href="https://twitter.com/dagworks" target="_blank">
    <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=X" alt="Follow DAGWorks on X">
  </a>
</div>

## Key Features

*   **State Machine Design:**  Express your AI applications as state machines, making complex logic manageable.
*   **Real-time Monitoring:**  Track, monitor, and trace your system's execution with the integrated UI.
*   **Flexible Integrations:** Connect with your preferred LLM frameworks, storage solutions, and observability tools.
*   **Easy-to-Use API:** Develop applications with simple Python functions.
*   **Comprehensive UI:** Built-in UI to visualize the execution and debug your state machines.
*   **Wide Range of Applications:** Build chatbots, agents, simulations, and more.

## Getting Started

1.  **Installation:**
    ```bash
    pip install "burr[start]"
    ```
    (See the [documentation](https://burr.dagworks.io/getting_started/install/) if you're using Poetry.)

2.  **Run the UI:**
    ```bash
    burr
    ```
    This opens the Burr UI, which comes with demo data and a chatbot example. You'll need to set the `OPENAI_API_KEY` environment variable to chat with the demo chatbot.

3.  **Explore the Examples:**
    ```bash
    git clone https://github.com/apache/burr && cd burr/examples/hello-world-counter
    python application.py
    ```
    See the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for more details.

## How Burr Works

Burr uses a simple, dependency-free Python library to build and manage state machines, providing a UI for introspection and debugging, and a set of integrations for persistence and connections.

The core API is simple -- the Burr hello-world looks like this (plug in your own LLM, or copy from [the docs](https://burr.dagworks.io/getting_started/simple-example/#build-a-simple-chatbot>) for _gpt-X_)

```python
from burr.core import action, State, ApplicationBuilder

@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    # your code -- write what you want here, for example
    chat_item = {"role" : "user", "content" : prompt}
    return state.update(prompt=prompt).append(chat_history=chat_item)

@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
    # query the LLM however you want (or don't use an LLM, up to you...)
    response = _query_llm(state["chat_history"]) # Burr doesn't care how you use LLMs!
    chat_item = {"role" : "system", "content" : response}
    return state.update(response=content).append(chat_history=chat_item)

app = (
    ApplicationBuilder()
    .with_actions(human_input, ai_response)
    .with_transitions(
        ("human_input", "ai_response"),
        ("ai_response", "human_input")
    ).with_state(chat_history=[])
    .with_entrypoint("human_input")
    .build()
)
*_, state = app.run(halt_after=["ai_response"], inputs={"prompt": "Who was Aaron Burr, sir?"})
print("answer:", app.state["response"])
```

## Use Cases

Burr is ideal for building:

*   Simple Chatbots
*   Stateful RAG-based Chatbots
*   LLM-based Adventure Games
*   Interactive Email Assistants
*   Simulations
*   Hyperparameter Tuning

## Comparison

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Join the Community

*   [Discord](https://discord.gg/6Zy2DwP4f3)
*   [Documentation](https://burr.dagworks.io/)
*   [Blog](https://blog.dagworks.io/p/burr-develop-stateful-ai-applications)

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Roadmap

*   FastAPI integration and hosted deployment
*   Core library improvements (retries, exception management, framework integrations, metadata)
*   Tooling for hosted execution
*   Additional storage integrations
*   Burr Cloud (sign up [here](https://forms.gle/w9u2QKcPrztApRedA))

```