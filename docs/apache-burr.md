# Burr: Build Stateful AI Applications with Ease

Burr is a Python library that simplifies the development of stateful AI applications, such as chatbots and agents, enabling you to build robust and easily-managed systems. [Visit the original repo](https://github.com/apache/burr)

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

## Key Features:

*   **Simplified State Management:** Express your application logic as a state machine for easier management and debugging.
*   **Real-time UI:** Monitor, track, and trace your system's execution in real-time using the built-in UI.
*   **Flexible Integrations:** Integrate with your favorite LLM frameworks, storage solutions, and other tools.
*   **Extensible Architecture:** Build custom actions and integrate with existing libraries like Hamilton.
*   **Easy to Get Started:** Quickly build and deploy with a straightforward Python API and clear documentation.

## Why Burr?

Burr empowers developers to build sophisticated AI applications, focusing on state management, decision-making, and persistent workflows.

## Quickstart:

1.  **Install:**

    ```bash
    pip install "burr[start]"
    ```
2.  **Run the UI:**

    ```bash
    burr
    ```
    This launches Burr's telemetry UI. Explore the demo chatbot application, which can be found under "Demos" in the sidebar.  You'll need to set the `OPENAI_API_KEY` environment variable.
3.  **Run an Example:**

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    View the counter example in your terminal and track the trace in the UI. For more details, see the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr lets you express your application as a state machine, allowing you to manage state, complex decisions, human feedback, and self-persisting workflows.

```python
from burr.core import action, State, ApplicationBuilder

@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    # your code
    chat_item = {"role" : "user", "content" : prompt}
    return state.update(prompt=prompt).append(chat_history=chat_item)

@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
    # query the LLM however you want
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

Burr includes:

1.  A dependency-free Python library.
2.  A UI for introspection and debugging.
3.  Integrations for state persistence, telemetry, and other systems.

## Applications of Burr:

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Interactive assistants (e.g., email writing)
*   Simulations and hyperparameter tuning (non-LLM use-cases)

## Comparison with Other Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Start Building

Refer to the [getting started](https://burr.dagworks.io/getting_started/simple-example) guide and explore the examples.

## Roadmap

*   FastAPI integration.
*   Efficiency and usability improvements.
*   Tooling for hosted execution and infrastructure integration.
*   Additional storage integrations.
*   Burr Cloud for hosted solutions (waitlist [here](https://forms.gle/w9u2QKcPrztApRedA)).

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
>
> **Ashish Ghosh**, *CTO, Peanut Robotics*
```
More testimonials from Reddit and other individuals are included in the original README.