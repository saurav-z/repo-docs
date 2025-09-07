# Burr: Build Stateful AI Applications with Ease

Burr is a Python library that simplifies the development of stateful AI applications, providing a robust framework for managing complex workflows.  ([Original Repo](https://github.com/apache/burr))

<div>

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

</div>

## Key Features

*   **State Machine Framework:** Define your application logic as a state machine for clear, manageable workflows.
*   **Real-time UI:** Monitor, trace, and debug your application's execution with a built-in UI.
*   **Flexible Integrations:** Integrate with your favorite LLMs, storage solutions, and other systems.
*   **Simple Python API:** Build state machines with easy-to-understand Python functions.
*   **Open Source:** Free and open-source with an active community.
*   **Wide Range of Use Cases:** Chatbots, agents, simulations, and more!

## Get Started Quickly

1.  **Install:** `pip install "burr[start]"`
2.  **Run the UI:** `burr`
3.  **Explore Demos:**  See the "chatbot" demo within the UI.  You'll need to set the `OPENAI_API_KEY` environment variable for full functionality.
4.  **Run an Example:**
    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```
    See the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for more details.

## How Burr Works

Burr uses a simple, dependency-free Python library to build and manage state machines. You define actions and transitions to create a flowchart of your application's logic.  It includes a UI to visualize execution telemetry.

### Example Code Snippet
```python
from burr.core import action, State, ApplicationBuilder

@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    chat_item = {"role" : "user", "content" : prompt}
    return state.update(prompt=prompt).append(chat_history=chat_item)

@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
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

## What Can You Build with Burr?

Burr is versatile and can be used for various applications, including:

*   Simple Chatbots
*   Stateful RAG-based Chatbots
*   LLM-based Adventure Games
*   Interactive Assistants (e.g., email writing)
*   Simulations and Hyperparameter Tuning
*   and much more!

## Comparison with Other Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why the name Burr?

Named after Aaron Burr, and connected to [Hamilton](https://github.com/dagworks-inc/hamilton), Burr represents a harmony between state management and dynamic execution within the DagWorks open-source ecosystem.

## Testimonials

*   (Quotes from the original README are included here)

## Roadmap

*   FastAPI Integration & Hosted Deployment
*   Core Library Improvements (retries, framework integrations, metadata)
*   Hosted Execution Tooling
*   Additional Storage Integrations

If you're interested in Burr Cloud, sign up for the waitlist [here](https://forms.gle/w9u2QKcPrztApRedA).

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Contributors

*   (List of contributors from original README)