# Burr: Build and Manage Stateful AI Applications with Ease

[![Discord](https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord)](https://discord.gg/6Zy2DwP4f3)
[![Downloads](https://static.pepy.tech/badge/burr/month)](https://pepy.tech/project/burr)
[![PyPI Downloads](https://static.pepy.tech/badge/burr)
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

**Burr is a powerful Python library that simplifies the development of stateful AI applications, offering a UI for real-time monitoring and debugging.**  [See the original repo](https://github.com/apache/burr).

## Key Features:

*   **State Machine Abstraction:** Express your application logic as a state machine, perfect for managing complex decisions and workflows.
*   **UI for Real-Time Monitoring:**  Track, monitor, and trace your system's execution with an intuitive user interface.
*   **Pluggable Persisters:** Save and load application state using various storage options.
*   **Framework-Agnostic:** Integrate seamlessly with your favorite LLMs, frameworks, and tools.
*   **Easy to Get Started:** Simple Python building blocks and a quick start guide to get you up and running fast.
*   **Extensible:** Build custom actions and integrations to suit your specific needs.

## Quick Start

Install Burr using pip:

```bash
pip install "burr[start]"
```

Run the UI server:

```bash
burr
```

Explore the demo chatbot application within the UI to see Burr in action (requires an OpenAI API key).

To run the counter example, clone the repo:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```
and see the trace being tracked in the UI.

Detailed instructions and examples are available in the [documentation](https://burr.dagworks.io/).

## How Burr Works

Burr's core is a simple yet powerful API that lets you define state machines using Python functions.  You define *actions* and *transitions* to control the flow of your application.

**Example**

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

## What Can You Build With Burr?

Burr supports a wide range of applications, including:

*   Simple chatbots and stateful RAG-based chatbots.
*   LLM-based adventure games and interactive assistants.
*   Simulations and hyperparameter tuning.

## Integrations and Extensibility

Burr offers:

*   **Integrations:** Connect with various vendors for LLM observability, storage, and more.
*   **Custom Actions:** Delegate to your favorite libraries like Hamilton.

## Comparison with Other Frameworks

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

*   FastAPI integration and hosted deployment.
*   Improved core library features (retries, exception management, etc.).
*   More integrations with popular frameworks.
*   Tooling for hosted execution.
*   Additional storage integrations.
*   Burr Cloud.

## Contributing

We welcome contributions! See the [developer docs](https://burr.dagworks.io/contributing) to get started.

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making." - *Ashish Ghosh, CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later." - *Reddit user cyan2k, LocalLlama, Subreddit*

(and more testimonials from various users)