# Burr: Build Powerful Stateful AI Applications

Burr is an open-source Python framework that simplifies the development of stateful AI applications, enabling you to build, monitor, and debug complex workflows with ease. Check out the original repo [here](https://github.com/apache/burr).

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

*   **State Machine Modeling:** Explicitly model your application logic as state machines.
*   **Framework Agnostic:** Works seamlessly with any of your favorite frameworks.
*   **Real-time UI:** Includes a user interface for monitoring, tracing, and debugging your applications.
*   **Pluggable Persisters:** Easily save and load application state.
*   **LLM Integration:** Designed for applications utilizing LLMs, but not limited to them.
*   **Idempotent Workflows:** Build workflows that are self-persisting.

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

(See the [documentation](https://burr.dagworks.io/getting_started/install/) if you're using poetry)

### Running the UI

Start the Burr UI server:

```bash
burr
```

This will launch the Burr telemetry UI. Explore the default data or the demo chatbot application (requires an OpenAI API key).

### Example

Clone the repository and run a simple example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

For more details, see the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr utilizes a core API that allows you to express your applications as state machines. This is done through simple python functions.
You can (and should!) use it for anything in which you have to manage state, track complex decisions, add human feedback, or dictate an idempotent, self-persisting workflow.

**Core Principles:**

*   **State Management:** Easily manage and track state changes within your applications.
*   **Action-Based Logic:** Build applications from simple Python functions, using the `@action` decorator to define actions.
*   **Transitions:** Define transitions between states to create a clear and manageable application flow.

## What You Can Build with Burr

Burr is ideal for a wide range of applications, including:

*   Chatbots (GPT-like, RAG-based, multi-modal)
*   LLM-based games
*   Interactive assistants (email writing, etc.)
*   Simulations and hyperparameter tuning

## Comparison with Other Frameworks

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

**Ashish Ghosh**
*CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."

**Reddit user cyan2k**
*LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."

**Ishita**
*Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."

**Matthew Rideout**
*Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."

**Rinat Gareev**
*Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."

**Hadi Nayebi**
*Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."

**Aditya K.**
*DS Architect, TaskHuman*

## Roadmap

*   FastAPI integration + hosted deployment.
*   Efficiency/usability improvements for the core library
    *   First-class support for retries + exception management
    *   More integration with popular frameworks (LCEL, LLamaIndex, Hamilton, etc...)
    *   Capturing & surfacing extra metadata.
    *   Improvements to the pydantic-based typing system
*   Tooling for hosted execution of state machines, integrating with your infrastructure
*   Additional storage integrations.

To get access to Burr Cloud, sign up for the waitlist [here](https://forms.gle/w9u2QKcPrztApRedA).

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Contributors

### Code Contributions

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