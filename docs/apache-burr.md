# Burr: Build and Monitor Stateful AI Applications

Burr empowers developers to create stateful AI applications with ease, providing a robust framework for managing complex workflows and integrating with your favorite tools.  [Explore the Burr repository on GitHub!](https://github.com/apache/burr)

## Key Features:

*   **State Machine Foundation:** Express your application logic as state machines for clear, manageable workflows.
*   **Framework Agnostic:** Integrate seamlessly with any LLM and your preferred frameworks.
*   **Real-Time UI:** Monitor, trace, and debug your applications with Burr's built-in user interface.
*   **Pluggable Persistence:** Save and load application state with flexible persister options.
*   **Extensible with Hooks:** Integrate with LLM providers, storage solutions, and other tools.
*   **Fast and Simple:** Easy to get started with a minimal Python library and a UI.

## Getting Started

### Installation:

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

### Run the UI:

```bash
burr
```

This will open the Burr UI, allowing you to visualize and interact with your state machines. The UI includes a demo chatbot example.

### Example Usage:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/apache/burr && cd burr/examples/hello-world-counter
    ```

2.  **Run the example:**

    ```bash
    python application.py
    ```

    See the counter example running in the terminal, along with the trace being tracked in the UI.

For more detailed information, consult the [Getting Started Guide](https://burr.dagworks.io/getting_started/simple-example/).

## Core Concepts

Burr uses a simple API to express applications as state machines:

*   **Actions:** Define functions that perform specific tasks.
*   **Transitions:** Define the flow between actions, creating a directed graph.
*   **State:**  Manage and track the application's current context.

Example:

```python
from burr.core import action, State, ApplicationBuilder

@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    # ...
    return state.update(prompt=prompt).append(chat_history=chat_item)

@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
    # ...
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

## Use Cases

Burr is ideal for a variety of AI applications:

*   Chatbots
*   RAG-based Chatbots
*   LLM-based adventure games
*   Email assistants
*   Simulations
*   Hyperparameter tuning

## Comparison

Burr distinguishes itself with its state machine approach, real-time UI, and framework-agnostic design.

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
>
> **Ashish Ghosh**, CTO, Peanut Robotics

(And more!)

## Roadmap

Future developments include:

*   FastAPI integration and hosted deployment.
*   Improvements to the core library (retries, exception management, etc.).
*   Enhanced integrations (LCEL, LLamaIndex, etc.)
*   Tooling for hosted execution.
*   Additional storage integrations.

## Contributing

We welcome contributions!  Find out how to get started with our [contributing guide](https://burr.dagworks.io/contributing).

## Contributors

(See the original README for a list of contributors)