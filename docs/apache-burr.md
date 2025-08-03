# Burr: Build Stateful AI Applications with Ease

**Burr empowers developers to build, monitor, and debug stateful AI applications like chatbots, agents, and simulations using Python.**

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

**Key Features:**

*   **State Machine Architecture:** Express your applications as state machines for clear logic and state management.
*   **Real-time UI:** Monitor, track, and debug your application's execution with Burr's built-in UI.
*   **Framework Agnostic:** Works with your favorite LLM frameworks and other tools.
*   **Pluggable Persisters:** Save and load application state with flexible persisters.
*   **Versatile Applications:** Build chatbots, agents, simulations, and more.

**Get Started Quickly:**

1.  **Installation:** `pip install "burr[start]"`
2.  **Run the UI:** `burr`
3.  **Explore Examples:** See the demo chatbot and other example applications.

**Learn More:**

*   [Documentation](https://burr.dagworks.io/)
*   [Quick Intro Video](https://www.loom.com/share/a10f163428b942fea55db1a84b1140d8?sid=1512863b-f533-4a42-a2f3-95b13deb07c9)
*   [Longer Video Intro & Walkthrough](https://www.youtube.com/watch?v=rEZ4oDN0GdU)
*   [Blog Post](https://blog.dagworks.io/p/burr-develop-stateful-ai-applications)

**[View the original repository on GitHub](https://github.com/apache/burr)**

## How Burr Works

Burr's core API allows you to define your application logic as a state machine.  Define actions with inputs, outputs, and transitions. The UI provides invaluable debugging insights and monitoring. Integrations make it easier to persist state, connect to telemetry, and integrate with other systems.

## What You Can Build with Burr

*   **Chatbots:** From simple GPT-like bots to stateful RAG applications.
*   **LLM-Based Games:** Create interactive adventure games.
*   **Assistants:** Build assistants for tasks like email writing.
*   **Simulations and More:** Explore diverse use cases beyond LLMs, like simulations and hyperparameter tuning.

## Comparison to Other Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why Burr?

Burr's name is inspired by Aaron Burr, referencing its connection to [Hamilton](https://github.com/dagworks-inc/hamilton), and the DAGWorks family of open-source projects. The name is a metaphor for bringing together disparate pieces to create a cohesive system.

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
>
> **Ashish Ghosh**, CTO, Peanut Robotics

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
>
> **Reddit user cyan2k**, LocalLlama, Subreddit

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
>
> **Ishita**, Founder, Watto.ai

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
>
> **Matthew Rideout**, Staff Software Engineer, Paxton AI

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
>
> **Rinat Gareev**, Senior Solutions Architect, Provectus

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
>
> **Hadi Nayebi**, Co-founder, CognitiveGraphs

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."
>
> **Aditya K.**, DS Architect, TaskHuman

## Roadmap

Planned features include:

1.  FastAPI integration and hosted deployment.
2.  Usability improvements (retries, exception management).
3.  More framework integrations (LCEL, LlamaIndex, Hamilton, etc.)
4.  Capture extra metadata for fine-tuning, etc.
5.  Improvements to the pydantic-based typing system.
6.  Tooling for hosted execution and infrastructure integration (Ray, Modal, etc.)
7.  Additional storage integrations.

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Contributors

### Code Contributions

-   [Elijah ben Izzy](https://github.com/elijahbenizzy)
-   [Stefan Krawczyk](https://github.com/skrawcz)
-   [Joseph Booth](https://github.com/jombooth)
-   [Nandani Thakur](https://github.com/NandaniThakur)
-   [Thierry Jean](https://github.com/zilto)
-   [Hamza Farhan](https://github.com/HamzaFarhan)
-   [Abdul Rafay](https://github.com/proftorch)
-   [Margaret Lange](https://github.com/margaretlange)

### Bug hunters/special mentions

-   [Luke Chadwick](https://github.com/vertis)
-   [Evans](https://github.com/sudoevans)
-   [Sasmitha Manathunga](https://github.com/mmz-001)