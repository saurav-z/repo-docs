# Burr: Build, Observe, and Scale Stateful AI Applications

Burr is a Python library that simplifies the development of stateful AI applications, offering a robust framework for building decision-making systems.  Find the original repo [here](https://github.com/apache/burr).

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

*   **State Machine Architecture:** Design and manage complex AI workflows with a state machine approach.
*   **Real-time UI:** Monitor, trace, and debug your applications with a user-friendly interface.
*   **Flexible Integrations:** Connect with your favorite LLMs, frameworks, and storage solutions.
*   **Framework Agnostic:**  Works seamlessly with various libraries like Langchain and Hamilton.
*   **Simplified Development:**  Develop applications using simple Python functions.
*   **Wide range of use cases:** Supports chatbots, agents, simulations, and more.

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

### Run the UI

Start the Burr UI server:

```bash
burr
```

This opens the Burr telemetry UI with example data, including a demo chatbot application.

### Example

Run a simple counter example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

See the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for more details.

## How Burr Works

Burr uses a state machine to represent your application logic. Define actions as Python functions and transitions between states.

Example:

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

1.  A Python library for building and managing state machines.
2.  A UI for execution telemetry.
3.  Integrations for state persistence and telemetry.

## Use Cases

Burr is ideal for:

*   Chatbots
*   RAG-based Chatbots
*   LLM-based adventure games
*   Interactive assistants
*   Simulations
*   Hyperparameter tuning

## Comparison

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Roadmap

Future developments include:

1.  FastAPI integration
2.  Efficiency/usability improvements
3.  Tooling for hosted execution
4.  Additional storage integrations

## Contributing

Contribute to Burr! See the [developer docs](https://burr.dagworks.io/contributing).

## Contributors

### Code Contributions
*   Elijah ben Izzy
*   Stefan Krawczyk
*   Joseph Booth
*   Nandani Thakur
*   Thierry Jean
*   Hamza Farhan
*   Abdul Rafay
*   Margaret Lange

### Bug hunters/special mentions
*   Luke Chadwick
*   Evans
*   Sasmitha Manathunga