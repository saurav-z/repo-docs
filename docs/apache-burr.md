# Burr: Build and Manage Stateful AI Applications with Ease

**Burr is a powerful open-source framework that simplifies the development of stateful AI applications, providing a robust solution for managing complex decision-making processes.**

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

*   **Effortless AI Application Development**: Burr makes it easy to build applications that make decisions from simple Python building blocks.
*   **Real-time Monitoring and Tracing**: Burr's built-in UI provides real-time visibility into your application's execution, aiding in debugging and understanding complex workflows.
*   **Flexible Integration**: Integrates with a wide range of LLMs and frameworks, making it easy to build on your existing technology stack.
*   **State Management**: Manage state, track complex decisions, add human feedback, and dictate an idempotent, self-persisting workflow.
*   **Pluggable Persisters**:  Save and load application state with pluggable persisters.

**[See the original repository for more details](https://github.com/apache/burr)**

## Key Features

*   **State Machine Architecture:** Build applications as state machines for clear, maintainable logic.
*   **Intuitive Python API:**  Uses a simple, dependency-free Python library.
*   **Telemetry UI:** A user-friendly UI for monitoring, tracing, and debugging.
*   **Extensible Integrations:** Easily integrate with your favorite LLM providers, observability tools, and storage solutions.

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

### Run the UI

```bash
burr
```

This will open the Burr telemetry UI, allowing you to explore pre-loaded data and demo applications.

### Example

Run the hello-world counter example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

See the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for further details.

## How Burr Works

Burr allows you to express your application as a state machine, with actions and transitions.

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

Burr combines a dependency-free Python library, a UI, and integrations for state persistence, telemetry, and external system connections.

![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## Use Cases

Burr can be applied to a wide variety of applications:

*   Simple chatbots
*   Stateful RAG-based chatbots
*   LLM-based adventure games
*   Email assistants
*   Simulations
*   Hyperparameter tuning

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

Named after Aaron Burr, the third VP of the United States, the project is a follow-up to the [Hamilton library](https://github.com/dagworks-inc/hamilton).

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
> **Ashish Ghosh**, CTO, Peanut Robotics

(plus 5 more testimonials)

## Roadmap

*   FastAPI integration + hosted deployment
*   First-class support for retries + exception management
*   More integration with popular frameworks (LCEL, LLamaIndex, Hamilton, etc...)
*   Capturing & surfacing extra metadata
*   Improvements to the pydantic-based typing system
*   Tooling for hosted execution of state machines
*   Additional storage integrations

## Contributing

See the [developer-facing docs](https://burr.dagworks.io/contributing) for information on contributing.

## Contributors

**(List of contributors)**