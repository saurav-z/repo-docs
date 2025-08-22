# Burr: Build and Manage Stateful AI Applications

Burr is the open-source framework that empowers you to build and manage stateful AI applications with ease, offering a powerful solution for complex decision-making processes.  ([View the original repository](https://github.com/apache/burr))

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

*   **State Machine Modeling:** Express your application logic as a state machine, simplifying complex workflows and decision-making processes.
*   **Real-Time UI:** Visualize execution telemetry and debug your applications with a built-in UI.
*   **Flexible Integrations:** Seamlessly integrate with your favorite LLMs, frameworks, and tools.
*   **Pluggable Persistence:** Save and load application state with customizable persisters.
*   **Framework-Agnostic:** Build upon simple Python building blocks, independent of external frameworks.

## Why Burr?

Burr allows you to:

*   Manage state effectively in your AI applications.
*   Track complex decisions and add human feedback.
*   Create idempotent and self-persisting workflows.
*   Scale your applications with ease.

## Getting Started

### Installation

```bash
pip install "burr[start]"
```

### Run the UI
```bash
burr
```

### Example

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

For a detailed explanation, see the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr uses a simple Python-based API for building and managing state machines.

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

## Applications of Burr

*   Chatbots
*   Stateful RAG-based Chatbots
*   LLM-based adventure games
*   Email assistant
*   Simulations
*   Hyperparameter Tuning

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

1.  FastAPI integration + hosted deployment
2.  First-class support for retries + exception management
3.  More integration with popular frameworks (LCEL, LLamaIndex, Hamilton, etc...)
4.  Capturing & surfacing extra metadata, e.g. annotations for particular point in time, that you can then pull out for fine-tuning, etc.
5.  Improvements to the pydantic-based typing system
6.  Tooling for hosted execution of state machines, integrating with your infrastructure (Ray, modal, FastAPI + EC2, etc...)
7.  Additional storage integrations. More integrations with technologies like MySQL, S3, etc. so you can run Burr on top of what you have available.

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
> **Ashish Ghosh**
> *CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
> **Reddit user cyan2k**
> *LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
> **Ishita**
> *Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
> **Matthew Rideout**
> *Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
> **Rinat Gareev**
> *Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
> **Hadi Nayebi**
> *Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."
> **Aditya K.**
> *DS Architect, TaskHuman*

## Contributing

We welcome contributions! To get started on developing, see the [developer-facing docs](https://burr.dagworks.io/contributing).

## Contributors

### Code contributions

Users who have contributed core functionality, integrations, or examples.

-   [Elijah ben Izzy](https://github.com/elijahbenizzy)
-   [Stefan Krawczyk](https://github.com/skrawcz)
-   [Joseph Booth](https://github.com/jombooth)
-   [Nandani Thakur](https://github.com/NandaniThakur)
-   [Thierry Jean](https://github.com/zilto)
-   [Hamza Farhan](https://github.com/HamzaFarhan)
-   [Abdul Rafay](https://github.com/proftorch)
-   [Margaret Lange](https://github.com/margaretlange)

### Bug hunters/special mentions

Users who have contributed small docs fixes, design suggestions, and found bugs

-   [Luke Chadwick](https://github.com/vertis)
-   [Evans](https://github.com/sudoevans)
-   [Sasmitha Manathunga](https://github.com/mmz-001)