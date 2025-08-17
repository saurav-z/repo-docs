# Burr: Build and Manage Stateful AI Applications with Ease

Burr simplifies the development of decision-making applications like chatbots, agents, and simulations, offering a powerful and flexible solution.  [Explore the Burr Repository](https://github.com/apache/burr)

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

Burr provides a robust framework to build and manage stateful AI applications, offering a user-friendly approach to state machine development.

**Key Features:**

*   **State Machine Framework:**  Build and manage your applications with a state machine, offering a clear and structured approach to development.
*   **Real-time UI:**  Monitor, track, and trace your system's execution in real-time using the intuitive UI.
*   **Pluggable Persisters:** Save and load application state with easy-to-integrate persisters.
*   **Flexible Integrations:** Integrate with your favorite LLM frameworks and other systems.
*   **Open Source:**  Benefit from the community and contribute to the project's growth.
*   **Easy to use:** Burr is easy to get started with and has a great developer experience.

**Quick Start**

1.  **Install:**

    ```bash
    pip install "burr[start]"
    ```

    (See the [installation instructions](https://burr.dagworks.io/getting_started/install/) if you're using poetry)
2.  **Run the UI:**

    ```bash
    burr
    ```

    This opens Burr's telemetry UI. Explore the demo chat application (requires `OPENAI_API_KEY`).

3.  **Code and Run Examples:**

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    See the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for more details.

**How Burr Works**

Express your application as a state machine (i.e. a graph/flowchart). The core API is simple -- a hello-world looks like this (plug in your own LLM, or copy from [the docs](https://burr.dagworks.io/getting_started/simple-example/#build-a-simple-chatbot>) for _gpt-X_)

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

Burr includes:

1.  A (dependency-free) low-abstraction python library that enables you to build and manage state machines with simple python functions
2.  A UI you can use view execution telemetry for introspection and debugging
3.  A set of integrations to make it easier to persist state, connect to telemetry, and integrate with other systems

![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

**What You Can Build with Burr**

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Interactive assistants (email writing)
*   Simulations (time-series forecasting)
*   Hyperparameter tuning and more!

**Comparison with other Frameworks**

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

**Why the Name Burr?**

Burr is named after Aaron Burr, the third Vice President of the United States. It shares a connection with the Hamilton library, another open-source project from DAGWorks.

**Testimonials**

(Quotes from satisfied users are included in original readme)

**Roadmap**

Burr has an active roadmap including:

*   FastAPI integration and hosted deployment.
*   Improvements for the core library (e.g., retries, framework integrations, metadata).
*   Tooling for hosted execution.
*   Additional storage integrations.
*   [Burr Cloud waitlist](https://forms.gle/w9u2QKcPrztApRedA)

**Contributing**

We welcome contributions!  See the [developer-facing docs](https://burr.dagworks.io/contributing).

**Contributors**

(List of contributors is included in original readme)