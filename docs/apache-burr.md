# Burr: Build and Monitor Stateful AI Applications with Ease

**Burr is an open-source framework that simplifies the development and monitoring of stateful AI applications by providing a simple, flexible state machine engine.**

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

[Explore the Burr GitHub Repository](https://github.com/apache/burr)

## Key Features of Burr:

*   **State Machine Engine:** Build applications as state machines with simple Python functions.
*   **Real-time UI:** Monitor, trace, and debug your system's execution in a user-friendly UI.
*   **Pluggable Persisters:** Save and load application state with flexible persistence options.
*   **Framework Agnostic:** Integrate seamlessly with your favorite LLM frameworks and tools.
*   **Comprehensive Examples:** Explore a variety of use cases, from chatbots to simulations.
*   **Open Source & Community Driven:** Benefit from a vibrant community and collaborative development.

## What can you do with Burr?

Burr is ideal for building a wide range of applications that require state management and complex decision-making, including:

*   Chatbots and Conversational AI
*   LLM-powered Agents
*   Simulations
*   Interactive Assistants
*   And much more!

## Quick Start

1.  **Install:** `pip install "burr[start]"`
2.  **Run the UI:** `burr`
3.  **Explore:** Access the UI and use the demo chatbot example or start running examples with `git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter && python application.py`
4.  **Code:** See the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr's core API enables you to express applications as state machines, managing state, tracking decisions, and enabling workflows. It provides a clear, concise approach to building complex AI systems.

## Comparison Against Common Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Testimonials

See what others are saying about Burr:

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
> **Ashish Ghosh**, *CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
> **Reddit user cyan2k**, *LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
> **Ishita**, *Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
> **Matthew Rideout**, *Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
> **Rinat Gareev**, *Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
> **Hadi Nayebi**, *Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."
> **Aditya K.**, *DS Architect, TaskHuman*

## Roadmap

*   FastAPI integration + hosted deployment
*   First-class support for retries + exception management
*   More integration with popular frameworks
*   Capturing & surfacing extra metadata
*   Improvements to the pydantic-based typing system
*   Tooling for hosted execution of state machines
*   Additional storage integrations

If you want to avoid self-hosting, sign up [here](https://forms.gle/w9u2QKcPrztApRedA) for the waitlist to get access to Burr Cloud.

## Contributing

Contribute to Burr's development!  See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Contributors

A list of all code contributors and bug finders is provided in the original documentation.
```

Key improvements and SEO considerations:

*   **Clear and Concise Hook:** The one-sentence summary at the beginning grabs attention and clearly states the value proposition.
*   **Keyword Optimization:** Includes relevant keywords like "state machine," "AI applications," "LLM," "framework," "monitoring," and "debugging" throughout the headings and content.
*   **Structured Headings:** Uses clear and informative headings to organize the information and improve readability.
*   **Bulleted Key Features:** Highlights the core benefits of Burr in an easy-to-scan format.
*   **Action-Oriented Language:** Uses verbs like "Build," "Monitor," and "Explore" to encourage engagement.
*   **Call to Action:** Provides clear steps for getting started and links to the documentation.
*   **Improved Readability:** Simplified the original content and removed some redundant phrases.
*   **Testimonials:** Kept the testimonials because social proof is essential.
*   **Roadmap & Contributing Sections:** Added for user engagement and developer relations.
*   **Links Back to the Repo** Maintained the original GitHub repository link at the beginning of the README.