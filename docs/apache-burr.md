# Burr: Build and Monitor State Machines for AI Applications

[![Discord](https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord)](https://discord.gg/6Zy2DwP4f3)
[![Downloads](https://static.pepy.tech/badge/burr/month)](https://pepy.tech/project/burr)
![PyPI Downloads](https://static.pepy.tech/badge/burr)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/dagworks-inc/burr)](https://github.com/dagworks-inc/burr/pulse)
[![X](https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social)](https://twitter.com/burr_framework)
<a target="_blank" href="https://linkedin.com/showcase/dagworks-inc" style="background:none">
  <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin" />
</a>
<a href="https://twitter.com/burr_framework" target="_blank">
  <img src="https://img.shields.io/badge/burr_framework-Follow-purple.svg?logo=X"/>
</a>
<a href="https://twitter.com/dagworks" target="_blank">
  <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=X"/>
</a>
<img src="https://github.com/user-attachments/assets/2ab9b499-7ca2-4ae9-af72-ccc775f30b4e" width=25 height=25/>

**Burr simplifies the creation and management of stateful AI applications, enabling you to build robust and traceable decision-making systems.**  [Explore the Burr Documentation](https://burr.dagworks.io/).

**Key Features:**

*   **State Machine Framework:** Design applications as state machines using simple Python functions.
*   **Real-time UI:** Monitor, trace, and debug your application's execution with a built-in UI.
*   **Flexible Integrations:** Connect with your favorite LLMs, storage solutions, and other systems.
*   **Simplified State Management:** Easily manage state, track complex decisions, and implement self-persisting workflows.
*   **Open Source:** A robust library designed to build production-ready AI applications.

**Getting Started:**

1.  **Installation:**

    ```bash
    pip install "burr[start]"
    ```

2.  **Run the UI:**

    ```bash
    burr
    ```

    This will launch the Burr telemetry UI.  Explore the demo chatbot to see Burr in action (requires `OPENAI_API_KEY`).

3.  **Example:** Clone and run a sample application:

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    View the counter example and trace it in the UI.

**How Burr Works:**

Burr enables you to define your application's logic as a state machine, making it easy to manage state, track decisions, and incorporate human feedback.

```python
from burr.core import action, State, ApplicationBuilder

@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    # ... your code ...
    return state.update(prompt=prompt).append(chat_history=chat_item)

@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
    # ... query the LLM ...
    response = _query_llm(state["chat_history"])
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
Burr comprises a dependency-free Python library and a UI for telemetry and debugging, facilitating seamless integration with various systems.

**What You Can Build with Burr:**

*   Simple and advanced chatbots (e.g., GPT-like, RAG-based)
*   LLM-powered adventure games
*   Email assistants
*   Simulations and hyperparameter tuning

**Comparison with Similar Frameworks:**

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

**Why Burr?**

Burr, named after Aaron Burr, offers a robust solution for building stateful AI applications, inspired by the need for state management in the Hamilton library.

**Testimonials:**

*   Quotes from users praising Burr's state management, UI, and ease of use. (See original README for details)

**Roadmap:**

Burr's development includes:

*   FastAPI integration and hosted deployment
*   Efficiency and usability improvements
*   Framework integrations (LCEL, LlamaIndex, etc.)
*   Hosted execution and storage integrations

**Contribute:**

We welcome contributions! See the [contributing guide](https://burr.dagworks.io/contributing) to get started.

**Contributors:**

*   (List of contributors from the original README)

**Find the original repo at:** [https://github.com/apache/burr](https://github.com/apache/burr)