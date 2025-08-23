# Burr: Build and Monitor State-Driven AI Applications with Ease

Burr empowers developers to build and monitor sophisticated, stateful AI applications using simple Python building blocks. [Visit the Burr GitHub Repository](https://github.com/apache/burr)

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

*   **State Machine-Based Development:** Define application logic as clear, manageable state machines using Python.
*   **Real-time UI Monitoring:** Track, monitor, and debug your AI applications through an intuitive user interface.
*   **Framework Agnostic:** Seamlessly integrate with your preferred LLM frameworks and libraries.
*   **Pluggable Persistence:** Easily save and load application state with pluggable persisters.
*   **Wide Range of Applications:** Suitable for chatbots, agents, simulations, and more.
*   **Open-Source:** Benefit from a collaborative open-source community.

## Getting Started

### Installation

Install Burr using pip:

```bash
pip install "burr[start]"
```

(See the [Burr Documentation](https://burr.dagworks.io/getting_started/install/) if you're using Poetry).

### Run the UI

Start the Burr UI server:

```bash
burr
```

This will launch the telemetry UI, preloaded with sample data and a demonstration chatbot.

### Quick Example

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

View the counter example running in the terminal and trace its execution in the UI.

### Documentation

Explore the [Burr Documentation](https://burr.dagworks.io/) for detailed guides and examples.

### Videos

*   Quick intro (<3min): [Burr Quick Intro](https://www.loom.com/share/a10f163428b942fea55db1a84b1140d8?sid=1512863b-f533-4a42-a2f3-95b13deb07c9)
*   Detailed Intro/Walkthrough: [Burr Video Intro & Walkthrough](https://www.youtube.com/watch?v=rEZ4oDN0GdU)

## How Burr Works

Burr simplifies application development by allowing you to express your application as a state machine.  You define actions and transitions, and Burr manages the state and execution flow.

### Core API Example

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

## What You Can Build With Burr

Burr supports a wide variety of applications, including:

*   Chatbots (GPT-like)
*   Stateful RAG-based Chatbots
*   LLM-based Adventure Games
*   Interactive Email Assistants
*   Simulations and Hyperparameter Tuning

## Comparison Table

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Roadmap

*   FastAPI integration & hosted deployment
*   Efficiency & Usability improvements
*   Tooling for hosted state machine execution
*   Additional storage integrations

## Contributing

We welcome contributions! Read the [contributing guidelines](https://burr.dagworks.io/contributing) to get started.

## Contributors

*   [Elijah ben Izzy](https://github.com/elijahbenizzy)
*   [Stefan Krawczyk](https://github.com/skrawcz)
*   [Joseph Booth](https://github.com/jombooth)
*   [Nandani Thakur](https://github.com/NandaniThakur)
*   [Thierry Jean](https://github.com/zilto)
*   [Hamza Farhan](https://github.com/HamzaFarhan)
*   [Abdul Rafay](https://github.com/proftorch)
*   [Margaret Lange](https://github.com/margaretlange)
*   [And Many More!](https://github.com/apache/burr/graphs/contributors)

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making." - **Ashish Ghosh, CTO, Peanut Robotics**

> "Honestly, take a look at Burr. Thank me later." - **Reddit user cyan2k, LocalLlama, Subreddit**

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top." - **Ishita, Founder, Watto.ai**

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI." - **Matthew Rideout, Staff Software Engineer, Paxton AI**

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that." - **Rinat Gareev, Senior Solutions Architect, Provectus**

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors." - **Hadi Nayebi, Co-founder, CognitiveGraphs**

> "Moving from LangChain to Burr was a game-changer! - **Aditya K., DS Architect, TaskHuman**