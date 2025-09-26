# Burr: Build and Monitor Stateful AI Applications

Burr simplifies developing stateful AI applications like chatbots, agents, and simulations using Python building blocks, offering a powerful and flexible way to manage complex decision-making processes.  Learn more about Burr's capabilities at the [original repository](https://github.com/apache/burr).

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

*   **State Machine-Based Design:**  Express your application logic as a state machine for clear, manageable workflows.
*   **Python-First:**  Build applications using familiar Python functions, making it easy to integrate with existing code.
*   **Real-Time UI:** Monitor and trace your application's execution with a built-in UI for debugging and understanding.
*   **Pluggable Persistence:** Integrate with various storage solutions to save and load application state.
*   **Framework Agnostic:**  Integrate with any LLM framework or other tools to build custom actions.

## Quick Start

1.  **Installation:** Install Burr using pip:
    ```bash
    pip install "burr[start]"
    ```
2.  **Run the UI:** Launch the Burr telemetry UI:
    ```bash
    burr
    ```
3.  **Explore Examples:**  Use the UI to explore example chatbots and other demos.  Then run the hello world counter example.
    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```
    See the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for more details.

## How Burr Works

Burr uses a simple API for creating state machines:

*   Define actions using Python functions.
*   Specify read and write operations for each action.
*   Define transitions between actions.
*   Build and run your application.

Burr includes a UI, and a set of integrations to persist state, connect to telemetry, and integrate with other systems.

[![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)](https://github.com/apache/burr)

## Use Cases

Burr can power diverse applications, including:

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Email assistants
*   Simulations
*   Hyperparameter tuning
*   And much more!

## Comparison with Other Frameworks

Burr offers a unique approach. Here's how it compares to other popular frameworks:

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why Burr?

Burr simplifies building complex AI applications, providing a modular, scalable, and observable framework.

## Roadmap

*   FastAPI integration and hosted deployment
*   Efficiency and usability improvements
*   Tooling for hosted state machine execution
*   Additional storage integrations

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) for details.

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
> **Ashish Ghosh, CTO, Peanut Robotics**

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
> **Reddit user cyan2k, LocalLlama, Subreddit**

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
> **Ishita, Founder, Watto.ai**

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
> **Matthew Rideout, Staff Software Engineer, Paxton AI**

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
> **Rinat Gareev, Senior Solutions Architect, Provectus**

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
> **Hadi Nayebi, Co-founder, CognitiveGraphs**

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."
> **Aditya K., DS Architect, TaskHuman**

## Contributors

*   [Elijah ben Izzy](https://github.com/elijahbenizzy)
*   [Stefan Krawczyk](https://github.com/skrawcz)
*   [Joseph Booth](https://github.com/jombooth)
*   [Nandani Thakur](https://github.com/NandaniThakur)
*   [Thierry Jean](https://github.com/zilto)
*   [Hamza Farhan](https://github.com/HamzaFarhan)
*   [Abdul Rafay](https://github.com/proftorch)
*   [Margaret Lange](https://github.com/margaretlange)

### Bug hunters/special mentions

*   [Luke Chadwick](https://github.com/vertis)
*   [Evans](https://github.com/sudoevans)
*   [Sasmitha Manathunga](https://github.com/mmz-001)