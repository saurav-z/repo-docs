# Burr: Build and Monitor Stateful AI Applications

Burr is a Python library that simplifies building and monitoring stateful AI applications, offering a UI for real-time tracing and debugging.  [Check out the original repo](https://github.com/apache/burr).

<div>

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

</div>

## Key Features

*   **State Machine Modeling:**  Define your application's logic as a state machine for clarity and control.
*   **Real-time UI:**  Monitor, trace, and debug your application's execution with a built-in UI.
*   **Framework Agnostic:** Integrate Burr with your favorite LLM and AI frameworks.
*   **Pluggable Persisters:** Save and load application state using various storage options.
*   **Easy Integration:**  Integrate with existing tools and build custom actions.
*   **Wide Variety of Use Cases:** Build and monitor stateful AI applications, including chatbots, agents, and simulations.

## Quick Start

1.  **Installation:** Install the Burr package using pip:

    ```bash
    pip install "burr[start]"
    ```

2.  **Run the UI:** Start the Burr telemetry UI:

    ```bash
    burr
    ```

3.  **Explore Examples:** Clone the repository and run example applications:

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    See the [documentation](https://burr.dagworks.io/getting_started/simple-example/) for more details.

## How Burr Works

Burr empowers you to model applications as state machines using simple Python functions.  You define actions and transitions to manage state, track decisions, and create self-persisting workflows.  Burr's core API is intuitive, allowing you to build complex AI applications with ease.

Burr provides a (dependency-free) low-abstraction python library, along with a UI you can use to view execution telemetry for introspection and debugging, and a set of integrations to make it easier to persist state, connect to telemetry, and integrate with other systems.

## Use Cases

Burr is ideal for:

*   GPT-like chatbots
*   Stateful RAG-based chatbots
*   LLM-based adventure games
*   Interactive assistants (e.g., for writing emails)
*   Simulations and hyperparameter tuning
*   And much more!

##  Comparison against Common Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

##  Why the name Burr?

Named after Aaron Burr, Burr is the second open-source library from DAGWorks, following Hamilton.

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making." - **Ashish Ghosh**, *CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later." - **Reddit user cyan2k**, *LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top." - **Ishita**, *Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI." - **Matthew Rideout**, *Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that." - **Rinat Gareev**, *Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors." - **Hadi Nayebi**, *Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since." - **Aditya K.**, *DS Architect, TaskHuman*

## Roadmap

Upcoming features include:

1.  FastAPI integration + hosted deployment
2.  Improvements to core library functionality (retries, integrations, metadata)
3.  Tooling for hosted state machine execution
4.  Additional storage integrations

Sign up for the Burr Cloud waitlist [here](https://forms.gle/w9u2QKcPrztApRedA).

## Contributing

We welcome contributions! Check out the [developer documentation](https://burr.dagworks.io/contributing) to get started.

## Contributors

### Code contributions

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