# Burr: Build Stateful AI Applications with Ease

Burr simplifies the development of stateful AI applications like chatbots, agents, and simulations, enabling developers to build and manage complex workflows with ease.

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

*   **State Machine Management:** Define and manage application logic as state machines for clarity and control.
*   **UI for Monitoring & Tracing:**  Visualize and debug your application's execution in real-time with Burr's built-in UI.
*   **Framework-Agnostic:** Integrate Burr with your preferred frameworks and tools.
*   **Extensible Architecture:**  Easily integrate with LLMs, storage solutions, and telemetry services.
*   **Idempotent Workflows:**  Ensure reliable and repeatable execution of your applications.

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

### Run the UI

Start the UI server:

```bash
burr
```

Access the telemetry UI. Explore the demo chatbot application to see how Burr's UI captures real-time updates.

### Example

Clone the repository and run a sample application:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```
Find the counter example's trace in the UI.  For more details see the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr allows you to represent your application as a state machine, managing state, tracking decisions, and ensuring idempotent workflows. The core API uses simple Python functions:

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

Burr provides a dependency-free Python library, a UI for introspection and debugging, and integrations for state persistence and telemetry.

![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## Use Cases

Burr empowers a variety of applications:

*   Chatbots
*   Stateful RAG-based chatbots
*   LLM-based adventure games
*   Interactive assistants (email writing)
*   Time-series forecasting simulations
*   Hyperparameter tuning and other machine-learning training

## Integrations

*   Integrate with your preferred vendors for LLM observability and storage.
*   Build custom actions that delegate to your favorite libraries, such as [Hamilton](https://github.com/DAGWorks-Inc/hamilton).

## Comparison with Other Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why "Burr"?

Named after Aaron Burr, the library complements [Hamilton](https://github.com/dagworks-inc/hamilton).  Burr was initially built to handle state management between executions of Hamilton DAGs.

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."

**Ashish Ghosh**
*CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."

**Reddit user cyan2k**
*LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."

**Ishita**
*Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."

**Matthew Rideout**
*Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."

**Rinat Gareev**
*Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."

**Hadi Nayebi**
*Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."

**Aditya K.**
*DS Architect, TaskHuman*

## Roadmap

Planned features include:

1.  FastAPI integration + hosted deployment.
2.  Usability improvements for the core library (retries, exception management, better integrations with frameworks like LCEL, LLamaIndex, and Hamilton).
3.  Tooling for hosted execution of state machines, integrating with your infrastructure (Ray, modal, FastAPI + EC2, etc...)
4.  Additional storage integrations.

Sign up for the [Burr Cloud waitlist](https://forms.gle/w9u2QKcPrztApRedA) if you are interested.

## Contributing

We welcome contributions! See the [developer documentation](https://burr.dagworks.io/contributing) for details.

## Contributors

See the original repository for a full list of [contributors](https://github.com/apache/burr).