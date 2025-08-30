# Burr: Build and Monitor Stateful AI Applications

Burr is an open-source Python library designed to help you develop stateful AI applications with ease.  [Explore Burr on GitHub](https://github.com/apache/burr).

<div>
    <a href="https://discord.gg/6Zy2DwP4f3" target="_blank">
        <img src="https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord" alt="Join Burr Discord"/>
    </a>
    <a href="https://pepy.tech/project/burr" target="_blank">
        <img src="https://static.pepy.tech/badge/burr/month" alt="Burr PyPI Downloads"/>
    </a>
    <img src="https://static.pepy.tech/badge/burr" alt="PyPI Downloads"/>
    <a href="https://github.com/dagworks-inc/burr/pulse" target="_blank">
        <img src="https://img.shields.io/github/last-commit/dagworks-inc/burr" alt="GitHub Last Commit"/>
    </a>
    <a href="https://twitter.com/burr_framework" target="_blank">
        <img src="https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social" alt="Follow Burr on X"/>
    </a>
    <a href="https://linkedin.com/showcase/dagworks-inc" target="_blank">
        <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin" alt="Follow DAGWorks on LinkedIn"/>
    </a>
    <a href="https://twitter.com/burr_framework" target="_blank">
        <img src="https://img.shields.io/badge/burr_framework-Follow-purple.svg?logo=X" alt="Follow Burr on X"/>
    </a>
    <a href="https://twitter.com/dagworks" target="_blank">
        <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=X" alt="Follow DAGWorks on X"/>
    </a>
</div>

Burr empowers developers to build robust and scalable AI applications, chatbots, agents, simulations, and more, by enabling state management and real-time monitoring.

## Key Features:

*   **State Machine Modeling:** Define your application logic as a state machine, making complex workflows manageable.
*   **Real-time Monitoring:** A user-friendly UI for tracking, debugging, and tracing your application's execution.
*   **Framework Agnostic:** Seamlessly integrates with your favorite AI frameworks and tools.
*   **Pluggable Persisters:** Save and load application state with various persistence options.
*   **Extensible:** Build custom actions and integrations to suit your specific needs.
*   **Open Source:** Benefit from a growing community and a transparent development process.

## Getting Started

Install Burr using pip:

```bash
pip install "burr[start]"
```

Run the UI server:

```bash
burr
```

Clone the example and run the counter:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

For detailed instructions, refer to the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr allows you to express applications as state machines, providing a core API that is easy to use.

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

Burr includes:

1.  A (dependency-free) low-abstraction python library.
2.  A UI you can use view execution telemetry for introspection and debugging
3.  A set of integrations to make it easier to persist state, connect to telemetry, and integrate with other systems

[![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## Use Cases

Burr can be applied to a wide range of AI-powered applications, including:

*   Chatbots
*   Stateful RAG-based chatbots
*   LLM-based adventure games
*   Interactive assistants

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

*   FastAPI integration
*   Efficiency/usability improvements
*   Tooling for hosted execution
*   Additional storage integrations.

## Contribute

We welcome contributions!  See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
>
> **Ashish Ghosh**
> *CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
>
> **Reddit user cyan2k**
> *LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
>
> **Ishita**
> *Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
>
> **Matthew Rideout**
> *Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
>
> **Rinat Gareev**
> *Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
>
> **Hadi Nayebi**
> *Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."
>
> **Aditya K.**
> *DS Architect, TaskHuman*

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