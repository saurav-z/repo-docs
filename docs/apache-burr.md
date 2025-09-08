# Burr: Build Stateful AI Applications with Ease

Burr empowers developers to create robust, stateful AI applications using simple Python building blocks, offering a flexible and intuitive framework for managing complex workflows. ([Original Repo](https://github.com/apache/burr))

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

*   **Simple State Machine Abstraction:** Define your application logic using a clear, understandable state machine approach.
*   **UI for Real-time Monitoring & Debugging:** Track, monitor, and trace your system's execution in real-time using the built-in UI.
*   **Pluggable Persistence:** Easily save and load application state with pluggable persisters for memory and other storage options.
*   **Framework-Agnostic Design:** Integrate with your favorite LLMs and frameworks.
*   **Extensive Integrations:**  Connect to telemetry, persist state, and build custom actions with ease.
*   **Versatile Applications:** Suitable for chatbots, agents, simulations, and various other applications that require state management.

## Quick Start

1.  **Installation:**

    ```bash
    pip install "burr[start]"
    ```

2.  **Run the UI:**

    ```bash
    burr
    ```

    This launches the Burr telemetry UI, which includes a demo chatbot to help you understand the framework. Requires `OPENAI_API_KEY` for full functionality, but the UI still functions without it.

3.  **Run an Example:**

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    The counter example will run in the terminal, and its trace will be visible in the UI.

4.  **Explore the Docs:**  For detailed instructions, refer to the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr models your application as a state machine, allowing you to:

*   Manage state effectively
*   Track complex decision-making processes
*   Incorporate human feedback
*   Ensure idempotent and self-persisting workflows

The core API involves defining actions (Python functions) that read and write to the application's state, and transitions between these actions.

Burr provides:

1.  A lightweight Python library for creating and managing state machines with simple Python functions.
2.  A UI for monitoring and debugging.
3.  Integrations for persistence, telemetry, and integration with other systems.

![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## What Can You Build With Burr?

Burr is perfect for a wide range of applications:

*   **Chatbots:**  Simple gpt-like chatbots, stateful RAG-based chatbots.
*   **Games:** LLM-based adventure games.
*   **Assistants:** Interactive assistants for tasks like writing emails.
*   **Simulations:** Time-series forecasting simulations.
*   **Machine Learning:** Hyperparameter tuning and other ML workflows.

Burr's integrations allow you to connect to your favorite tools and frameworks, like [Hamilton](https://github.com/DAGWorks-Inc/hamilton).

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

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
> **Ashish Ghosh** - *CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
> **Reddit user cyan2k** - *LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
> **Ishita** - *Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
> **Matthew Rideout** - *Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
> **Rinat Gareev** - *Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
> **Hadi Nayebi** - *Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."
> **Aditya K.** - *DS Architect, TaskHuman*

## Roadmap

Future plans include:

1.  **FastAPI integration + hosted deployment:** to streamline production app deployment.
2.  **Core Library Improvements:** Retries, exception management, and integration with popular frameworks.
3.  **Hosted Execution Tooling:** Integrations with infrastructure (Ray, Modal, FastAPI + EC2, etc.).
4.  **Additional Storage Integrations:** Support for MySQL, S3, and other storage solutions.

Sign up for the [Burr Cloud waitlist](https://forms.gle/w9u2QKcPrztApRedA) for early access.

## Contributing

We welcome contributions! See the [developer documentation](https://burr.dagworks.io/contributing) for details.

## Contributors

*   **Code Contributions:** [List of contributors]
*   **Bug Hunters/Special Mentions:** [List of contributors]