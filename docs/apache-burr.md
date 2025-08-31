# Burr: Build and Monitor Stateful AI Applications (LLMs, Chatbots, & More)

Burr empowers developers to build and monitor stateful AI applications with ease, using simple Python building blocks.  [Check out the original repo](https://github.com/apache/burr).

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


## Key Features

*   **State Machine Framework:** Build applications as state machines using simple Python functions.
*   **Real-time UI:** Track, monitor, and trace application execution with a built-in UI.
*   **Flexible Integrations:** Easily integrate with your favorite LLM frameworks, data storage, and observability tools.
*   **Pluggable Persisters:** Save and load application state.
*   **Idempotent Workflows:** Manage state, track decisions, and handle feedback with ease.
*   **Broad Applicability:** Perfect for chatbots, agents, simulations, and any application managing complex state.

## Quickstart

1.  **Install:** `pip install "burr[start]"`
2.  **Run UI:** `burr`
3.  **Explore Examples:** See the chatbot demo in the UI, or run a simple example:

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    See [the docs](https://burr.dagworks.io/getting_started/install/) for more info.

## How Burr Works

Burr lets you express applications as state machines (graphs/flowcharts). The core API is simple, as the following "hello world" example shows:

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

Burr provides:

1.  A dependency-free Python library.
2.  A UI for execution telemetry.
3.  Integrations for state persistence and system integration.

![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## What Can You Build with Burr?

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Email assistants
*   Simulations
*   Hyperparameter tuning
*   And much more!

Integrations and custom actions allow you to integrate with LLM observability, storage, and delegate to other libraries. Burr helps you tie everything together in a way that scales.

## Comparison with Other Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why the Name "Burr"?

Named after Aaron Burr, and built by the team behind the [Hamilton library](https://github.com/dagworks-inc/hamilton). Burr and Hamilton represent a vision of harmony in state management.

## Testimonials

*Quotes from satisfied users - see original README*

## Roadmap

1.  FastAPI integration + hosted deployment
2.  Efficiency & usability improvements
3.  Tooling for hosted execution of state machines
4.  Additional storage integrations

Sign up for the Burr Cloud waitlist [here](https://forms.gle/w9u2QKcPrztApRedA).

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Contributors

*List of contributors - see original README*