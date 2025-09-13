# Burr: Build and Manage Stateful AI Applications with Ease

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

[Burr](https://github.com/apache/burr) empowers developers to build robust and scalable stateful AI applications like chatbots and agents by simplifying complex decision-making processes with easy-to-use Python building blocks.

## Key Features

*   **State Machine Architecture:** Express your applications as state machines (graphs/flowcharts) for clear logic and control.
*   **Real-time UI:** Track, monitor, and trace your system's execution with a built-in user interface.
*   **Pluggable Persistence:** Save and load application state using pluggable persisters.
*   **Framework Agnostic:** Integrate with your favorite LLMs and frameworks seamlessly.
*   **Easy to Use:** Simple Python functions for building state machines.

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

### Run the UI

Run the UI server:

```bash
burr
```

This opens the Burr telemetry UI, pre-loaded with demo data. Explore the "Demos" sidebar and select `chatbot`.

### Example

Clone the repository and run a simple example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

See the [documentation](https://burr.dagworks.io/) for detailed instructions and examples.

## How Burr Works

Burr lets you define your application as a state machine, where you have actions and transitions between these.

*   **Actions:** Defined with simple Python functions.
*   **Transitions:** Define the flow of execution.

Burr includes:

1.  A dependency-free Python library.
2.  A UI for real-time execution telemetry.
3.  Integrations for state persistence, telemetry, and integration with other systems.

![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## Use Cases

Burr is versatile and can be applied to various AI applications:

*   Simple Chatbots
*   Stateful RAG Chatbots
*   LLM-based Adventure Games
*   Interactive Assistants (e.g., for writing emails)
*   Simulations
*   Hyperparameter Tuning

## Comparisons with Similar Frameworks

Burr offers unique advantages in state management and UI for monitoring/tracing.

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why the Name?

Burr is named after Aaron Burr, in reference to DAGWorks' first open-source library, Hamilton, with which Burr was initially designed to support.

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making." - **Ashish Ghosh, CTO, Peanut Robotics**

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later." - **Reddit user cyan2k, LocalLlama, Subreddit**

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top." - **Ishita, Founder, Watto.ai**

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI." - **Matthew Rideout, Staff Software Engineer, Paxton AI**

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that." - **Rinat Gareev, Senior Solutions Architect, Provectus**

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors." - **Hadi Nayebi, Co-founder, CognitiveGraphs**

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since." - **Aditya K., DS Architect, TaskHuman**

## Roadmap

*   FastAPI integration and hosted deployment.
*   Core library improvements (retries, exception management, integrations, metadata).
*   Tooling for hosted execution and infrastructure integration.
*   Additional storage integrations.
*   Burr Cloud - sign up [here](https://forms.gle/w9u2QKcPrztApRedA) for the waitlist.

## Contributing

We welcome contributors! Refer to the [developer-facing docs](https://burr.dagworks.io/contributing).

## Contributors

### Code Contributions

-   Elijah ben Izzy
-   Stefan Krawczyk
-   Joseph Booth
-   Nandani Thakur
-   Thierry Jean
-   Hamza Farhan
-   Abdul Rafay
-   Margaret Lange

### Bug Hunters/Special Mentions

-   Luke Chadwick
-   Evans
-   Sasmitha Manathunga