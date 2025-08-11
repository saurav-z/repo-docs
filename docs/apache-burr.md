# Burr: Build and Manage State Machines for AI Applications

Burr empowers you to create robust, stateful AI applications, like chatbots and agents, with ease.

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

**Burr is a Python library that simplifies the development of stateful AI applications by providing a simple, flexible, and observable framework.**

## Key Features

*   **State Machine Design:** Define your application logic as state machines, making complex workflows manageable.
*   **Easy Integration:** Works seamlessly with your favorite LLMs and frameworks.
*   **Real-time UI:** Visualize, monitor, and debug your applications with Burr's built-in UI.
*   **Pluggable Persistence:** Save and load application state using a variety of storage options.
*   **Framework-Agnostic:** Leverage a core Python library that integrates with diverse systems.

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

(See the [documentation](https://burr.dagworks.io/getting_started/install/) for poetry installations.)

### Run the UI

Start the Burr telemetry UI:

```bash
burr
```

Explore the UI, which comes preloaded with demo data, including a chatbot.  The chatbot uses the `OPENAI_API_KEY` environment variable to interact, but you can still observe the UI's real-time state tracking without it.

### Example: Hello World Counter

1.  Clone the Burr repository:

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    ```

2.  Run the example:

    ```bash
    python application.py
    ```

Observe the counter's trace in the terminal and in the UI.  For a more comprehensive overview, refer to the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr utilizes a state machine model, allowing you to represent your application as a graph. This is ideal for managing state, tracking decisions, and implementing workflows that are self-persisting and idempotent.

Key API elements:

*   **Actions:** Python functions that represent operations within your state machine.
*   **State:**  Manages the data used by your application.
*   **Transitions:** Define the flow between actions.

Example Hello World (simple chatbot):

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

Burr provides a library to build and manage state machines, a UI for debugging, and integrations for persistence and connecting to various systems.

[![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)]

## Applications of Burr

Burr excels in building a variety of AI applications:

*   Simple Chatbots
*   Stateful RAG-based Chatbots
*   LLM-based Adventure Games
*   Interactive Email Assistants
*   Simulations (e.g., time-series forecasting)
*   Hyperparameter Tuning

Burr provides hooks and integrations to incorporate any vendor or library.

## Comparing Burr

Burr differentiates itself with its unique focus on state machine management and its UI for monitoring/tracing.

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why the Name "Burr"?

Burr is named after Aaron Burr, referencing DAGWorks' previous library, Hamilton, and their historical connection, reflecting the framework's use for connecting different technologies.

## Testimonials

(Include existing testimonials here, formatted if necessary.)

## Roadmap

Burr's future development includes:

1.  FastAPI integration and hosted deployment.
2.  Improvements to the core library, including:

    *   First-class support for retries and exception management.
    *   More integrations with popular frameworks.
    *   Improvements to pydantic-based typing.
3.  Tooling for hosted execution of state machines.
4.  Additional storage integrations.

If you are interested in Burr Cloud, sign up [here](https://forms.gle/w9u2QKcPrztApRedA) to be on the waitlist!

## Contributing

Contributions are welcome!  See the [developer-facing docs](https://burr.dagworks.io/contributing) for instructions.

## Contributors

(Include existing contributor information here.)

**Explore the full potential of Burr on [GitHub](https://github.com/apache/burr) and start building your next-generation AI application!**