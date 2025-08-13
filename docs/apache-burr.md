# Burr: Build Stateful AI Applications with Ease

Burr simplifies the development of stateful AI applications, providing a robust framework for building chatbots, agents, and simulations.  Find the original repository on [GitHub](https://github.com/apache/burr).

<div>
  <!-- Badges - original repo already has them -->
</div>

**Key Features:**

*   **State Machine Abstraction:** Build and manage complex workflows using a simple Python-based state machine approach.
*   **Real-time UI:** Visualize and debug your application's execution with a built-in telemetry UI.
*   **Flexible Integration:**  Integrate with your favorite LLMs, frameworks, and tools.
*   **Pluggable Persistence:** Easily save and load application state with various storage options.
*   **Open Source:** Benefit from the community and customize the project to fit your specific needs.

## Getting Started

### Installation

Install Burr using pip:

```bash
pip install "burr[start]"
```

### Run the UI

```bash
burr
```

Access the telemetry UI at `http://localhost:8000` (default).  Explore the demo chatbot application after setting the `OPENAI_API_KEY` environment variable or explore without key.

### Example:  Hello World Counter

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    ```

2.  **Run the example:**

    ```bash
    python application.py
    ```

    Observe the counter in the terminal and trace its execution within the UI.  Refer to the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for detailed instructions.

## How Burr Works

Burr allows you to structure applications as state machines. Define actions as Python functions that read and write state.  Transitions between actions define the application flow.

**Core API Example:**

```python
from burr.core import action, State, ApplicationBuilder

@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    # ... (your code) ...
    return state.update(prompt=prompt).append(chat_history=chat_item)

@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
    # ... (LLM query) ...
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

Burr can be used to build applications like:

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Email assistants
*   Simulations
*   Hyperparameter tuning

## Comparison with Similar Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Roadmap

*   FastAPI Integration and Hosted Deployment
*   Core Library Enhancements (retries, exception handling, integrations)
*   Tooling for Hosted Execution
*   Additional Storage Integrations

## Contributing

We welcome contributions! Refer to the [developer-facing docs](https://burr.dagworks.io/contributing) for details.

## Testimonials

> "[...] elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
>
> **Ashish Ghosh**, *CTO, Peanut Robotics*

(More testimonials are included in the original README)

## Contributors

(List of contributors from original README)