# Burr: Build and Monitor Stateful AI Applications

Burr is a Python library that simplifies the development of stateful AI applications like chatbots and agents using a simple state machine approach. **[Explore Burr on GitHub!](https://github.com/apache/burr)**

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

*   **State Machine Framework:** Build and manage complex workflows with ease using a simple, Pythonic state machine API.
*   **Real-time UI:** Monitor, trace, and debug your AI applications with Burr's built-in telemetry UI.
*   **Pluggable Persistence:** Save and load application state with flexible persister integrations.
*   **Framework Agnostic:** Burr seamlessly integrates with your favorite LLM frameworks and tools.
*   **Versatile Use Cases:** Suitable for chatbots, agents, simulations, and more.

## Getting Started

Install Burr:

```bash
pip install "burr[start]"
```

Run the UI:

```bash
burr
```

Explore the UI and demo chatbot app. Then, dive into the [documentation](https://burr.dagworks.io/) and [quickstart examples](https://github.com/dagworks-inc/burr).

## How Burr Works

Burr represents your application as a state machine. You define actions (functions) and transitions between them, creating a clear and manageable workflow.  Here's a simplified example:

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

Burr provides a dependency-free Python library, a UI for monitoring, and integrations for persistence and system integration.

## Use Cases

Burr excels in a variety of applications:

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Interactive assistants (email writing)
*   Simulations
*   Hyperparameter tuning

Integrate with your favorite vendors and libraries like [Hamilton](https://github.com/DAGWorks-Inc/hamilton).

## Comparison

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making." - **Ashish Ghosh, CTO, Peanut Robotics**

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later." - **Reddit user cyan2k, LocalLlama, Subreddit**

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top." - **Ishita, Founder, Watto.ai**

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI." - **Matthew Rideout, Staff Software Engineer, Paxton AI**

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that." - **Rinat Gareev, Senior Solutions Architect, Provectus**

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors." - **Hadi Nayebi, Co-founder, CognitiveGraphs**

> "Moving from LangChain to Burr was a game-changer! - **Aditya K., DS Architect, TaskHuman**

## Roadmap

Explore upcoming features like:

1.  FastAPI integration
2.  Efficiency and usability improvements
3.  Hosted execution tooling
4.  Expanded storage integrations

Sign up for the waitlist for [Burr Cloud](https://forms.gle/w9u2QKcPrztApRedA).

## Contributing

Contribute to Burr! Check out the [developer docs](https://burr.dagworks.io/contributing).

## Contributors

Thank you to all code contributors and bug hunters!