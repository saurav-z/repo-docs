# Burr: Build Stateful AI Applications with Ease

Burr is a powerful, open-source framework that simplifies the development of stateful AI applications, offering intuitive state management, a user-friendly UI, and flexible integrations.  [Explore the Burr GitHub repository](https://github.com/apache/burr)!

<div>
  <!-- Shields and Badges -->
  <a href="https://discord.gg/6Zy2DwP4f3">
    <img src="https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord" alt="Join Burr Discord"/>
  </a>
  <a href="https://pepy.tech/project/burr">
    <img src="https://static.pepy.tech/badge/burr/month" alt="PyPI Downloads (monthly)"/>
  </a>
  <img src="https://static.pepy.tech/badge/burr" alt="PyPI Downloads"/>
  <a href="https://github.com/dagworks-inc/burr/pulse">
    <img src="https://img.shields.io/github/last-commit/dagworks-inc/burr" alt="GitHub Last Commit"/>
  </a>
  <a href="https://twitter.com/burr_framework" style="background:none">
    <img src="https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social" alt="Follow on X (Twitter)"/>
  </a>
  <a href="https://linkedin.com/showcase/dagworks-inc" style="background:none">
    <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin" alt="Follow DAGWorks on LinkedIn"/>
  </a>
</div>

## Key Features:

*   **Simplified State Management:** Define and manage application state using Python functions.
*   **Real-time UI:** Monitor, trace, and debug your state machine executions with a built-in UI.
*   **Flexible Integrations:** Integrate with your favorite LLMs, frameworks, and storage solutions.
*   **Framework Agnostic:** Works seamlessly with various libraries and tools.
*   **Built-in Telemetry:** Track execution telemetry for introspection and debugging.
*   **Idempotent Workflows:** Design self-persisting workflows that add human feedback or complex decisions.

## Getting Started

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

Run the UI server:

```bash
burr
```

See [the documentation](https://burr.dagworks.io/) for more detailed instructions and examples.

## How Burr Works

Burr allows you to express your application as a state machine, utilizing a simple API with Python functions.

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

## Use Cases

Burr can be used to create a wide range of applications:

*   Chatbots (GPT-like and RAG-based)
*   LLM-based adventure games
*   Interactive assistants
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

Future improvements include:

*   FastAPI integration and hosted deployment
*   Enhanced core library features (retries, exception management)
*   More integrations with popular frameworks
*   Tooling for hosted execution
*   Additional storage integrations

Sign up for the waitlist for Burr Cloud [here](https://forms.gle/w9u2QKcPrztApRedA).

## Contributing

We welcome contributions!  See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
>
> **Ashish Ghosh**, *CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
>
> **Reddit user cyan2k**, *LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
>
> **Ishita**, *Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
>
> **Matthew Rideout**, *Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
>
> **Rinat Gareev**, *Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
>
> **Hadi Nayebi**, *Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."
>
> **Aditya K.**, *DS Architect, TaskHuman*

## Contributors
### Code contributions

Users who have contributed core functionality, integrations, or examples.

- [Elijah ben Izzy](https://github.com/elijahbenizzy)
- [Stefan Krawczyk](https://github.com/skrawcz)
- [Joseph Booth](https://github.com/jombooth)
- [Nandani Thakur](https://github.com/NandaniThakur)
- [Thierry Jean](https://github.com/zilto)
- [Hamza Farhan](https://github.com/HamzaFarhan)
- [Abdul Rafay](https://github.com/proftorch)
- [Margaret Lange](https://github.com/margaretlange)

### Bug hunters/special mentions

Users who have contributed small docs fixes, design suggestions, and found bugs

- [Luke Chadwick](https://github.com/vertis)
- [Evans](https://github.com/sudoevans)
- [Sasmitha Manathunga](https://github.com/mmz-001)