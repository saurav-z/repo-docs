# Burr: Build, Monitor, and Debug Stateful AI Applications

Burr is a Python library that simplifies the development of stateful AI applications like chatbots, agents, and simulations, offering a flexible framework for building and managing complex workflows. ([View on GitHub](https://github.com/apache/burr))

<div>
  <!-- Badges -->
  <a href="https://discord.gg/6Zy2DwP4f3">
    <img src="https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord" alt="Join Burr Discord">
  </a>
  <a href="https://pepy.tech/project/burr">
    <img src="https://static.pepy.tech/badge/burr/month" alt="Monthly Downloads">
  </a>
  <img src="https://static.pepy.tech/badge/burr" alt="PyPI Downloads">
  <a href="https://github.com/dagworks-inc/burr/pulse">
    <img src="https://img.shields.io/github/last-commit/dagworks-inc/burr" alt="Last Commit">
  </a>
  <a href="https://twitter.com/burr_framework">
    <img src="https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social" alt="Follow on X (Twitter)">
  </a>
  <a href="https://www.linkedin.com/showcase/dagworks-inc">
    <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin" alt="Follow DAGWorks on LinkedIn">
  </a>
  <a href="https://twitter.com/burr_framework">
    <img src="https://img.shields.io/badge/burr_framework-Follow-purple.svg?logo=X" alt="Follow Burr on X">
  </a>
  <a href="https://twitter.com/dagworks">
    <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=X" alt="Follow DAGWorks on X">
  </a>
</div>

## Key Features

*   **State Machine Abstraction:** Express your application logic as a state machine, simplifying complex workflows.
*   **Real-time Monitoring & Debugging:** Includes a UI to track, monitor, and trace your system's execution in real-time.
*   **Flexible Integrations:** Works with popular frameworks and offers pluggable persisters for saving and loading application state.
*   **Dependency-Free Core:**  A low-abstraction Python library for building and managing state machines.
*   **Wide Range of Use Cases:** Suitable for chatbots, agents, simulations, and other applications requiring state management.

## Quick Start

Install Burr using pip:

```bash
pip install "burr[start]"
```

Run the UI server:

```bash
burr
```

Explore the UI, including a demo chatbot.  To use the chatbot, set the `OPENAI_API_KEY` environment variable.

Then, try a simple example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

For detailed installation and usage, see the [getting started guide](https://burr.dagworks.io/getting_started/install/) and the [documentation](https://burr.dagworks.io/).

## How Burr Works

Burr lets you define your application as a state machine.  You define actions (Python functions) and transitions between them.

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

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Interactive assistants (e.g., email writing)
*   Simulations (time-series forecasting)
*   Hyperparameter tuning

## Comparison with Other Frameworks

Burr offers a unique approach by explicitly modeling state machines and providing a UI for monitoring, with a focus on framework-agnostic design.

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Roadmap

*   FastAPI integration and hosted deployment
*   Core library improvements (retries, exception handling, integrations)
*   Tooling for hosted execution
*   Additional storage integrations
*   Burr Cloud (waitlist available)

## Contributing

Contributions are welcome! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

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

*   [Elijah ben Izzy](https://github.com/elijahbenizzy)
*   [Stefan Krawczyk](https://github.com/skrawcz)
*   [Joseph Booth](https://github.com/jombooth)
*   [Nandani Thakur](https://github.com/NandaniThakur)
*   [Thierry Jean](https://github.com/zilto)
*   [Hamza Farhan](https://github.com/HamzaFarhan)
*   [Abdul Rafay](https://github.com/proftorch)
*   [Margaret Lange](https://github.com/margaretlange)

### Bug Hunters/Special Mentions
*   [Luke Chadwick](https://github.com/vertis)
*   [Evans](https://github.com/sudoevans)
*   [Sasmitha Manathunga](https://github.com/mmz-001)