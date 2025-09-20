# Burr: Build and Monitor Stateful AI Applications

Burr is a Python library that simplifies the creation and management of stateful AI applications, offering a clear path to building robust and observable systems.  ([View on GitHub](https://github.com/apache/burr))

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

*   **State Machine Framework:** Express your applications as state machines for clear, manageable logic.
*   **UI for Observability:**  Real-time monitoring and tracing of your system's execution with a built-in UI.
*   **Framework-Agnostic:** Integrates seamlessly with your favorite LLMs, frameworks, and tools.
*   **Pluggable Persistence:** Save and load application state with flexible persister options.
*   **Simple Python API:**  Build your state machines using straightforward Python functions.
*   **Built-in Examples:** Ready-to-use examples for chatbots, RAG applications, and more.

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

(See the [documentation](https://burr.dagworks.io/getting_started/install/) if you're using Poetry).

### Running the UI

Start the Burr UI server:

```bash
burr
```

The UI provides real-time telemetry, including a demo chatbot.  You'll need to set the `OPENAI_API_KEY` environment variable to test the full functionality of the chatbot.

### Example

Clone the repository and run a simple counter example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

Find the trace of your example in the UI.  See the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for more details.

## How Burr Works

Burr lets you build applications as state machines, ideal for managing state, tracking complex decisions, and incorporating feedback.  Here's a basic example:

```python
from burr.core import action, State, ApplicationBuilder

@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    chat_item = {"role" : "user", "content" : prompt}
    return state.update(prompt=prompt).append(chat_history=chat_item)

@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
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

1.  A lightweight Python library to build and manage state machines.
2.  A UI for execution telemetry, introspection, and debugging.
3.  Integrations for state persistence, telemetry, and other systems.

![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## Use Cases

Burr supports various applications:

1.  [Simple GPT-like chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/multi-modal-chatbot)
2.  [Stateful RAG-based chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/conversational-rag/simple_example)
3.  [LLM-based adventure game](https://github.com/DAGWorks-Inc/burr/tree/main/examples/llm-adventure-game)
4.  [Interactive assistant for writing emails](https://github.com/DAGWorks-Inc/burr/tree/main/examples/email-assistant)

It also supports non-LLM use cases like [simulation](https://github.com/DAGWorks-Inc/burr/tree/main/examples/simulation) and [hyperparameter tuning](https://github.com/DAGWorks-Inc/burr/tree/main/examples/ml-training).

Burr helps you scale your AI applications.

## Comparison with Other Frameworks

| Criteria                                          | Burr | Langgraph | Temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Contributing

We welcome contributions!  See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Roadmap

Future plans for Burr include:

1.  FastAPI integration & hosted deployment
2.  Core library improvements (retries, exception management, integrations, metadata)
3.  Tooling for hosted execution
4.  Additional storage integrations

Sign up for the Burr Cloud waitlist [here](https://forms.gle/w9u2QKcPrztApRedA).

## Testimonials

[Include Testimonials as listed in the original README]