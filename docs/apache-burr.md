# Burr: Build Stateful AI Applications with Ease

Burr is a powerful Python library designed to simplify the development of stateful AI applications, offering a streamlined approach to building decision-making systems like chatbots, agents, and simulations.  [Explore the Burr repository here!](https://github.com/apache/burr)

<br>
[![Discord](https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord)](https://discord.gg/6Zy2DwP4f3)
[![Downloads](https://static.pepy.tech/badge/burr/month)](https://pepy.tech/project/burr)
![PyPI Downloads](https://static.pepy.tech/badge/burr)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/apache/burr)](https://github.com/apache/burr/pulse)
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

<br>

## Key Features:

*   **State Machine Abstraction:**  Express your applications as state machines for clear logic and easy management.
*   **Real-time UI:**  Monitor, track, and debug your system's execution with a built-in telemetry UI.
*   **Simple Python API:**  Build and manage state machines using a clean, dependency-free Python library.
*   **Pluggable Persisters:** Save and load application state with flexible storage options.
*   **Flexible Integration:**  Integrate seamlessly with your favorite LLMs and other frameworks.
*   **Diverse Use Cases:**  Suitable for chatbots, agents, simulations, and various decision-making applications.

## Getting Started

### Installation:

Install Burr from PyPI:

```bash
pip install "burr[start]"
```
*(See [the documentation](https://burr.dagworks.io/getting_started/install/) if you're using poetry)*

### Run the UI:

```bash
burr
```

This opens Burr's telemetry UI, allowing you to explore example data and a demo chatbot.  Set the `OPENAI_API_KEY` environment variable to interact with the chatbot.

### Run a simple example

```bash
git clone https://github.com/apache/burr && cd burr/examples/hello-world-counter
python application.py
```

For more details, refer to the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr enables you to define your application as a state machine.  You define actions (functions) that modify state, and transitions between these actions.  Burr's UI helps you visualize and understand this state flow.

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

## Use Cases

Burr supports a variety of applications, including:

*   Simple chatbots
*   Stateful RAG-based chatbots
*   LLM-based adventure games
*   Interactive email assistants
*   Simulations
*   Hyperparameter tuning

## Comparison with Other Frameworks

Burr offers a unique approach to building stateful AI applications, offering explicit state management and UI-based monitoring.  Here's a comparison:

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Roadmap

Future enhancements include:

*   FastAPI integration & hosted deployment
*   Core library improvements (retries, exception handling, framework integrations, metadata)
*   Tooling for hosted execution
*   Additional storage integrations
*   Burr Cloud for hosted solutions (sign up for the waitlist [here](https://forms.gle/w9u2QKcPrztApRedA))

## Contributing

Contributions are welcome!  See the [developer documentation](https://burr.dagworks.io/contributing) to get started.

## Testimonials

*(Include testimonials from the original README here)*

## Contributors

*(Include the list of contributors from the original README here)*