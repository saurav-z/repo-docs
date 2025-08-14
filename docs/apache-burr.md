# Burr: Build Stateful AI Applications with Ease

Burr empowers developers to build stateful AI applications like chatbots, agents, and simulations using simple Python building blocks.

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

[**View the original repository on GitHub**](https://github.com/apache/burr)

## Key Features

*   **State Machine Modeling:** Express your application logic as a state machine for clarity and control.
*   **UI for Monitoring & Tracing:** Visualize execution telemetry in real-time to debug and understand your application's behavior.
*   **Pluggable Persisters:** Easily save and load application state with integrations for various storage solutions.
*   **Framework Agnostic:** Works seamlessly with your favorite frameworks and tools.
*   **Extensible:** Integrate with LLMs, observability tools, storage solutions, and custom actions.
*   **Use Cases:** Supports a wide range of applications, including chatbots, agents, simulations, and more.

## Getting Started

Get up and running quickly with Burr:

1.  **Installation:**
    ```bash
    pip install "burr[start]"
    ```
2.  **Run the UI:**
    ```bash
    burr
    ```
    This launches the Burr telemetry UI.
3.  **Explore Examples:** Clone the repository and run the hello-world counter example:

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```
    Then view the trace in the UI.

    For comprehensive installation instructions, see the [documentation](https://burr.dagworks.io/getting_started/install/).

## How Burr Works

Burr enables you to build state machines using simple Python functions. The core API is designed to be easy to use and understand.
Burr includes:

1.  A Python library to build and manage state machines with simple Python functions
2.  A UI you can use view execution telemetry for introspection and debugging
3.  A set of integrations to make it easier to persist state, connect to telemetry, and integrate with other systems

## Use Cases

Burr is versatile and can be used to build a variety of AI applications, including:

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Interactive assistants (email writing)
*   Simulations (time-series forecasting)
*   Hyperparameter tuning

## Comparison with Other Frameworks

Burr offers a unique approach compared to other popular frameworks:

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why Burr?

Burr is named after Aaron Burr, the third Vice President of the United States, and the namesake of Hamilton's nemesis, Alexander Hamilton. Burr was originally built as a harness to handle state between executions of Hamilton DAGs, but then realized that it had a wide array of applications and decided to release it more broadly.

## Testimonials

Read what others are saying about Burr:

*   **Ashish Ghosh**, CTO, Peanut Robotics: "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
*   **Reddit user cyan2k**, LocalLlama, Subreddit: "Honestly, take a look at Burr. Thank me later."
*   **Ishita**, Founder, Watto.ai: "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
*   **Matthew Rideout**, Staff Software Engineer, Paxton AI: "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
*   **Rinat Gareev**, Senior Solutions Architect, Provectus: "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
*   **Hadi Nayebi**, Co-founder, CognitiveGraphs: "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
*   **Aditya K.**, DS Architect, TaskHuman: "Moving from LangChain to Burr was a game-changer!"

## Roadmap

Burr is constantly evolving, with these upcoming features planned:

1.  FastAPI integration + hosted deployment
2.  Efficiency and usability improvements for the core library
3.  Tooling for hosted execution of state machines
4.  Additional storage integrations

Sign up for the Burr Cloud waitlist [here](https://forms.gle/w9u2QKcPrztApRedA).

## Contributing

We welcome contributions! Learn how to contribute by reviewing the [developer-facing docs](https://burr.dagworks.io/contributing).

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