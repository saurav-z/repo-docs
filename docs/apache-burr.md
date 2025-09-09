# Burr: Build, Monitor, and Scale Stateful AI Applications

Burr simplifies the creation of intelligent applications like chatbots and agents with its flexible state machine approach. [See the original repo](https://github.com/apache/burr)

<div>
  <!-- Badges -->
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

*   **State Machine Foundation:** Structure your AI applications as state machines, offering clear control flow and management.
*   **Real-time Monitoring & Tracing:** A built-in UI provides real-time insights, debugging, and telemetry for your application.
*   **Flexible Integrations:** Easily integrate with LLMs, data stores, and other third-party services, vendors and frameworks.
*   **Python-First Approach:** Develop with a simple, dependency-free core Python library.
*   **Idempotent Workflows:** Design applications that are self-persisting and maintain their state, even after interruptions.

## Getting Started

Install Burr using pip:

```bash
pip install "burr[start]"
```

Then launch the UI:

```bash
burr
```

Explore the UI, including the demo chatbot. The examples will help you get started.

*   Quick Start Guide: [https://burr.dagworks.io/getting\_started/simple-example/](https://burr.dagworks.io/getting_started/simple-example/)
*   Documentation: [https://burr.dagworks.io/](https://burr.dagworks.io/)
*   Demo Video: [https://www.loom.com/share/a10f163428b942fea55db1a84b1140d8?sid=1512863b-f533-4a42-a2f3-95b13deb07c9](https://www.loom.com/share/a10f163428b942fea55db1a84b1140d8?sid=1512863b-f533-4a42-a2f3-95b13deb07c9)

## How Burr Works

Burr enables stateful application development by allowing you to represent your application as a state machine. This structured approach aids in:

*   Managing state effectively
*   Tracking complex decision-making processes
*   Incorporating user feedback seamlessly
*   Creating self-persisting workflows

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

Burr is well-suited for building a range of AI-powered applications, including:

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Email assistants
*   Simulations and hyperparameter tuning

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

Burr is continuously evolving with exciting new features, including:

1.  FastAPI integration and hosted deployment.
2.  Improvements in the core library and integrations.
3.  Tooling for hosted execution of state machines.
4.  Additional storage integrations.

Learn more about upcoming features and sign up for the waitlist for Burr Cloud [here](https://forms.gle/w9u2QKcPrztApRedA).

## Contributing

We welcome contributions! See the [developer docs](https://burr.dagworks.io/contributing) to get started.

## Community

*   Join the Burr Discord: [https://discord.gg/6Zy2DwP4f3](https://discord.gg/6Zy2DwP4f3)

## Testimonials

*   Read what users are saying about Burr's impact:

    *   Ashish Ghosh, CTO, Peanut Robotics
    *   Reddit User cyan2k, LocalLlama, Subreddit
    *   Ishita, Founder, Watto.ai
    *   Matthew Rideout, Staff Software Engineer, Paxton AI
    *   Rinat Gareev, Senior Solutions Architect, Provectus
    *   Hadi Nayebi, Co-founder, CognitiveGraphs
    *   Aditya K., DS Architect, TaskHuman

## Contributors

*   Elijah ben Izzy
*   Stefan Krawczyk
*   Joseph Booth
*   Nandani Thakur
*   Thierry Jean
*   Hamza Farhan
*   Abdul Rafay
*   Margaret Lange
*   Luke Chadwick
*   Evans
*   Sasmitha Manathunga