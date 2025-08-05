# Burr: Build State-Driven AI Applications with Ease

**Burr empowers developers to create stateful AI applications like chatbots, agents, and simulations using simple Python building blocks.**

[View the original repository on GitHub](https://github.com/apache/burr)

## Key Features:

*   **State Machine Core:** Build and manage complex workflows using a clear, concise, and dependency-free Python library.
*   **Real-time UI:** Visualize, monitor, and debug your application's execution with a user-friendly interface.
*   **Flexible Integrations:** Seamlessly integrate with your favorite LLM frameworks, storage solutions, and telemetry tools.
*   **Versatile Use Cases:** Suitable for a wide range of applications, including chatbots, RAG systems, LLM-based games, and more.
*   **Idempotent Workflows:** Ensures reliable and repeatable execution of your AI applications.

## Getting Started

### Installation

Install Burr using pip:

```bash
pip install "burr[start]"
```

(See the [documentation](https://burr.dagworks.io/getting_started/install/) for more installation options.)

### Run the UI

Start the Burr UI server:

```bash
burr
```

Access the UI in your browser to explore example data and a demo chatbot.

### Example

Clone the repository and run a simple example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

For a detailed guide, see the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr uses a state machine approach to model your application's logic. You define actions and transitions, creating a clear and manageable workflow.

**Core Concepts:**

*   Define actions using simple Python functions.
*   Specify read and write operations to manage the application's state.
*   Define transitions to determine the flow between actions.
*   The included UI is a real-time way to visualize how state changes during execution.

**Example Code Snippet:**

```python
from burr.core import action, State, ApplicationBuilder

@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    chat_item = {"role" : "user", "content" : prompt}
    return state.update(prompt=prompt).append(chat_history=chat_item)

@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
    response = _query_llm(state["chat_history"])
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

Burr is well-suited for:

*   Chatbots
*   RAG-based Chatbots
*   LLM-based Games
*   Interactive Assistants
*   Simulations
*   Hyperparameter Tuning

## Comparison Against Similar Frameworks

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
> **Ashish Ghosh** - *CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
> **Reddit user cyan2k** - *LocalLlama, Subreddit*

## Roadmap

*   FastAPI integration for easy deployment.
*   Enhancements for core library usability and efficiency (retries, exception management, etc.).
*   Improved integrations with popular frameworks.
*   Tooling for hosted state machine execution.
*   Additional storage integrations.
*   Burr Cloud. To let us know you're interested sign up [here](https://forms.gle/w9u2QKcPrztApRedA) for the waitlist to get access.

## Contributing

Contributions are welcome! Check out the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Contributors

See the original README for a complete list of contributors.