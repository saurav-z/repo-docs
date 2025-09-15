# Burr: Build Stateful AI Applications with Ease

**Burr is an open-source Python framework designed to simplify the development of stateful AI applications like chatbots, agents, and simulations.** [Explore the Burr repository on GitHub](https://github.com/apache/burr).

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

*   **State Machine Design:** Define applications as state machines with simple Python functions.
*   **Real-time UI:** Track, monitor, and trace your system's execution in a user-friendly interface.
*   **Pluggable Persisters:** Save and load application state with customizable persisters.
*   **Framework-Agnostic:** Integrate with your favorite frameworks and tools.
*   **Comprehensive Integrations:** Seamlessly connect with telemetry, storage, and other systems.

## Quick Start

1.  **Install:**

    ```bash
    pip install "burr[start]"
    ```

2.  **Run UI Server:**

    ```bash
    burr
    ```

    This opens the Burr telemetry UI, which comes with demo data and a chatbot demo (requires `OPENAI_API_KEY`).

3.  **Run a simple example:**

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    See [documentation](https://burr.dagworks.io/) and video intro: [here](https://www.loom.com/share/a10f163428b942fea55db1a84b1140d8?sid=1512863b-f533-4a42-a2f3-95b13deb07c9).

## How Burr Works

Burr helps you build and manage state machines, track complex decisions, and integrate with other services.

*   **Core API:** Simple, dependency-free Python library.
*   **UI:** Visualize execution telemetry for debugging and introspection.
*   **Integrations:** Easily persist state, connect to telemetry, and integrate with other systems.

[Example code](https://burr.dagworks.io/getting_started/simple-example/#build-a-simple-chatbot)

## What You Can Build with Burr

Burr is versatile, powering:

*   Chatbots (GPT-like and RAG-based)
*   LLM-based adventure games
*   Interactive assistants (e.g., email writers)
*   Simulations and hyperparameter tuning
*   And more!

## Comparison to Other Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Roadmap

*   FastAPI integration and hosted deployment
*   Efficiency and usability improvements
*   Hosted execution tooling (Ray, Modal, FastAPI, etc.)
*   Additional storage integrations
*   [Burr Cloud waitlist](https://forms.gle/w9u2QKcPrztApRedA)

## Contributing

We welcome contributors! See [developer docs](https://burr.dagworks.io/contributing) for details.

## Community

*   [Join the Burr Discord](https://discord.gg/6Zy2DwP4f3)

## Testimonials

> _"After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."_
**Ashish Ghosh**, *CTO, Peanut Robotics*

> _"Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."_
**Reddit user cyan2k**, *LocalLlama, Subreddit*

(more testimonials)