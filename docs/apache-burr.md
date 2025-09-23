# Burr: Build Stateful AI Applications with Ease

Burr is a powerful Python framework that simplifies the development of stateful AI applications, offering a streamlined approach to managing complex workflows and decision-making processes. ([Original Repo](https://github.com/apache/burr))

<div>
    <a href="https://discord.gg/6Zy2DwP4f3">
        <img src="https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord" alt="Join Burr Discord" />
    </a>
    <a href="https://pepy.tech/project/burr">
        <img src="https://static.pepy.tech/badge/burr/month" alt="PyPI Downloads"/>
    </a>
    <img src="https://static.pepy.tech/badge/burr" alt="PyPI Downloads" />
    <a href="https://github.com/dagworks-inc/burr/pulse">
        <img src="https://img.shields.io/github/last-commit/dagworks-inc/burr" alt="GitHub Last Commit" />
    </a>
    <a href="https://twitter.com/burr_framework">
        <img src="https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social" alt="Follow @burr_framework on X" />
    </a>
    <a href="https://linkedin.com/showcase/dagworks-inc" style="background:none">
        <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin" alt="Follow DAGWorks on LinkedIn" />
    </a>
    <a href="https://twitter.com/burr_framework" target="_blank">
        <img src="https://img.shields.io/badge/burr_framework-Follow-purple.svg?logo=X" alt="Follow Burr Framework on X"/>
    </a>
    <a href="https://twitter.com/dagworks" target="_blank">
        <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=X" alt="Follow DAGWorks on X"/>
    </a>
</div>

## Key Features

*   **State Machine Design:** Express your application logic as a state machine for clear and organized workflows.
*   **Intuitive Python API:** Utilize a simple and easy-to-learn Python API for building and managing state machines.
*   **Real-time UI:** Benefit from a built-in UI to track, monitor, and debug your system's execution.
*   **Pluggable Persisters:** Save and load application state using flexible and customizable persisters.
*   **Framework Agnostic:** Seamlessly integrates with your favorite frameworks and libraries.
*   **Versatile Applications:** Suitable for chatbots, agents, simulations, and other decision-making applications.

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

This will open the Burr telemetry UI.  The UI comes preloaded with demo data and a chatbot demo for easy exploration.

### Example

Here's a simple "Hello World" example using the core Burr API.

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

## How Burr Works

Burr empowers developers to create stateful applications by expressing logic as state machines. Its core components include:

*   **Python Library:** A low-abstraction, dependency-free library for building and managing state machines using simple Python functions.
*   **UI:** A user-friendly UI for execution telemetry, introspection, and debugging.
*   **Integrations:** A set of integrations for state persistence, telemetry, and other system connections.

## Use Cases

Burr can be used to build:

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Email assistants
*   Simulations
*   Hyperparameter tuning
*   And more!

## Comparison with Other Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."

**Ashish Ghosh**
*CTO, Peanut Robotics*

And many more!

## Roadmap

*   FastAPI integration + hosted deployment
*   Improvements to the core library
*   Tooling for hosted execution
*   Additional storage integrations

## Contributing

We welcome contributions!  See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Contributors

*   Elijah ben Izzy
*   Stefan Krawczyk
*   Joseph Booth
*   Nandani Thakur
*   Thierry Jean
*   Hamza Farhan
*   Abdul Rafay
*   Margaret Lange
*   And more!