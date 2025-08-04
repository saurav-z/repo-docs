# Burr: Build Powerful, Stateful AI Applications with Ease

Burr is a Python library that simplifies the development of stateful AI applications, enabling you to build and manage complex workflows for chatbots, agents, simulations, and more.  [Explore Burr on GitHub](https://github.com/apache/burr)!

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

*   **Simple State Machine Abstraction:** Define your application logic using easy-to-understand state machines with Python functions.
*   **Real-time UI:**  Track, monitor, and trace your application's execution in a dedicated UI.
*   **Pluggable Persisters:**  Save and load application state with flexible persister options (e.g., for memory).
*   **Framework Agnostic:** Integrate with your favorite LLM frameworks and tools.
*   **Debugging & Introspection:** Easily debug and understand complex workflows with the UI.
*   **Extensible & Customizable:** Build custom actions and integrate with other systems to suit your needs.

## Getting Started

### Installation

Install Burr using pip:

```bash
pip install "burr[start]"
```

(See the [documentation](https://burr.dagworks.io/getting_started/install/) if you are using Poetry.)

### Run the UI

Start the Burr UI:

```bash
burr
```

This will open the telemetry UI. Explore the demo chatbot application by selecting "Demos" and then "chatbot". Ensure you have your `OPENAI_API_KEY` environment variable set for full functionality.

### Example

Clone the example repository and run the hello-world-counter:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

This will demonstrate the counter example running and its trace in the UI. For more details, see the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr allows you to express your applications as state machines.  You define actions (Python functions) that modify state, and transitions that control the flow of execution.

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

Burr's key components include:

1.  **Python Library:** A lightweight, dependency-free library for building and managing state machines.
2.  **UI:** A UI for introspection, debugging, and monitoring execution.
3.  **Integrations:**  Built-in integrations for state persistence, telemetry, and connection with other systems.

[See the Example GIF](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif) to see Burr at work!

## Use Cases

Burr empowers a variety of AI applications, including:

1.  [Simple GPT-like Chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/multi-modal-chatbot)
2.  [Stateful RAG-based Chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/conversational-rag/simple_example)
3.  [LLM-based Adventure Game](https://github.com/DAGWorks-Inc/burr/tree/main/examples/llm-adventure-game)
4.  [Interactive Email Assistant](https://github.com/dagworks-inc/burr/tree/main/examples/email-assistant)
5.  And more, including simulations and hyperparameter tuning examples.

## Comparison to Other Frameworks

Burr offers a unique approach, with the following differences:

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why the Name "Burr"?

Burr is named after Aaron Burr, reflecting a connection to the open-source library [Hamilton](https://github.com/dagworks-inc/hamilton). Burr was originally built to handle state between executions of Hamilton DAGs.

## Testimonials

*(Testimonials from original README)*

## Roadmap

Burr development includes:

1.  FastAPI integration + hosted deployment
2.  Usability improvements (retries, exception management, integrations)
3.  Hosted execution tooling
4.  Additional storage integrations

Sign up for the Burr Cloud waitlist [here](https://forms.gle/w9u2QKcPrztApRedA).

## Contributing

Contributions are welcome!  Refer to the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Contributors

*(Contributors from original README)*