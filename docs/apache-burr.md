# Burr: Build Stateful AI Applications with Ease

Burr is a Python framework that simplifies the development of stateful AI applications, providing a clear and manageable way to build decision-making systems. 

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

**Key Features:**

*   **State Machine Modeling:** Express your application logic as a state machine for clear, maintainable code.
*   **Real-time UI:**  Visualize and debug your application's execution with a built-in telemetry UI.
*   **Framework-Agnostic:** Integrates seamlessly with your favorite Python frameworks and LLMs.
*   **Pluggable Persisters:** Save and load application state using various storage options.
*   **Extensible Actions:** Build custom actions that integrate with other libraries (like Hamilton).
*   **Open Source:** Completely open-source, empowering you to inspect and use the source code.

**Get Started Quickly:**

1.  **Installation:**
    ```bash
    pip install "burr[start]"
    ```
2.  **Run the UI:**
    ```bash
    burr
    ```
    This will launch the Burr telemetry UI where you can explore example demos.
3.  **Explore Examples:**
    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```
    See how the counter example runs and is traced in the UI.

**Learn More:**

*   [Documentation](https://burr.dagworks.io/)
*   [Quick Intro Video](https://www.loom.com/share/a10f163428b942fea55db1a84b1140d8?sid=1512863b-f533-4a42-a2f3-95b13deb07c9)
*   [Detailed Video Walkthrough](https://www.youtube.com/watch?v=rEZ4oDN0GdU)
*   [Blog Post](https://blog.dagworks.io/p/burr-develop-stateful-ai-applications)
*   [Join the Discord Community](https://discord.gg/6Zy2DwP4f3)

**How Burr Works:**

Burr's core API allows you to define actions (functions) that modify the state of your application. You then define transitions between these actions, creating a state machine that governs the flow of your application.  Burr handles the state management, tracking, and persistence, allowing you to focus on the core logic.

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

**Use Cases:**

Burr is ideal for building:

*   GPT-like Chatbots
*   Stateful RAG-based Chatbots
*   LLM-based Adventure Games
*   Interactive Assistants (e.g., email writing)
*   And more!  Including simulations and hyperparameter tuning.

**Comparison to Other Frameworks:**

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

**Why Burr?**

Named after Aaron Burr, Burr is designed to bring harmony to complex AI application development. Built by the team behind [Hamilton](https://github.com/dagworks-inc/hamilton), Burr helps you manage state and workflow, simplifying the integration of various components.

**Testimonials**

[Includes the testimonials from the original README]

**Roadmap**

[Includes the roadmap from the original README]

**Contribute**

We welcome contributions!  See the [developer-facing docs](https://burr.dagworks.io/contributing) for guidance.

**Contributors**

[Includes contributor list from the original README]

**Visit the original repository for more details:** [https://github.com/apache/burr](https://github.com/apache/burr)