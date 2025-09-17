# Burr: Build and Monitor Stateful AI Applications with Ease

Burr is a powerful Python library that simplifies the development of stateful AI applications like chatbots, agents, and simulations, offering a clear and manageable way to build complex workflows.  [Explore the Burr repository on GitHub](https://github.com/apache/burr)!

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

*   **State Machine Modeling:**  Define application logic as state machines for clarity and control.
*   **Real-time UI:** Track, monitor, and trace your application's execution with a built-in UI.
*   **Pluggable Persistence:**  Save and load application state with flexible persisters.
*   **Framework Agnostic:** Integrate Burr with your preferred LLM frameworks and tools.
*   **Simplified Development:** Build AI applications from Python building blocks.

## Why Burr?

Burr empowers developers to build robust and observable AI-powered applications by managing state, complex decisions, and workflows.  It's especially useful for:

*   Chatbots and Conversational AI
*   AI Agents and Automation
*   Simulations and Decision-Making Systems
*   Any application requiring state management and workflow orchestration

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

### Run the UI

```bash
burr
```

The UI provides real-time insights into your application's execution.

### Run an Example

Clone the repository and run a simple example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

[See the documentation for detailed getting started guides](https://burr.dagworks.io/).

## How Burr Works

Burr uses a simple yet powerful API to define your application as a state machine.

*   Define actions using the `@action` decorator.
*   Specify reads and writes for each action.
*   Build an application with actions, transitions, and an initial state.
*   Run the application and monitor its execution in the UI.

## What Can You Build?

Burr is versatile and can be used for a variety of applications:

*   Simple and advanced chatbots.
*   Stateful RAG-based chatbots.
*   LLM-powered adventure games.
*   Email assistants.
*   Simulations and ML training.

## Comparison with other frameworks

Burr offers a unique combination of features. Here's a comparison:

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Roadmap

Future developments for Burr include:

1.  FastAPI integration + hosted deployment.
2.  Efficiency and usability improvements.
3.  Tooling for hosted execution.
4.  Additional storage integrations.

## Contributing

Contributions are welcome!  See the [developer-facing docs](https://burr.dagworks.io/contributing) for details.

## Testimonials
> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
**Ashish Ghosh, CTO, Peanut Robotics**

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
**Reddit user cyan2k, LocalLlama, Subreddit**

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
**Ishita, Founder, Watto.ai**

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
**Matthew Rideout, Staff Software Engineer, Paxton AI**

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
**Rinat Gareev, Senior Solutions Architect, Provectus**

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
**Hadi Nayebi, Co-founder, CognitiveGraphs**

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."
**Aditya K., DS Architect, TaskHuman**
## Contributors

### Code contributions
- [Elijah ben Izzy](https://github.com/elijahbenizzy)
- [Stefan Krawczyk](https://github.com/skrawcz)
- [Joseph Booth](https://github.com/jombooth)
- [Nandani Thakur](https://github.com/NandaniThakur)
- [Thierry Jean](https://github.com/zilto)
- [Hamza Farhan](https://github.com/HamzaFarhan)
- [Abdul Rafay](https://github.com/proftorch)
- [Margaret Lange](https://github.com/margaretlange)

### Bug hunters/special mentions

- [Luke Chadwick](https://github.com/vertis)
- [Evans](https://github.com/sudoevans)
- [Sasmitha Manathunga](https://github.com/mmz-001)