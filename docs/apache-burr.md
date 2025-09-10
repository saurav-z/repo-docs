# Burr: Build and Monitor Stateful AI Applications

Burr empowers developers to build robust, stateful AI applications with simple Python building blocks, offering a powerful solution for managing complex workflows and decisions.  [Visit the original repository](https://github.com/apache/burr).

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

*   **State Machine Framework:**  Express your application logic as state machines, perfect for managing complex flows and decisions.
*   **Real-time UI:**  A built-in UI provides real-time monitoring, tracing, and debugging capabilities for your applications.
*   **Framework Agnostic:** Burr seamlessly integrates with your preferred LLMs, frameworks, and tools.
*   **Pluggable Persistence:** Save and load application state with flexible storage options.
*   **Versatile Use Cases:** Suitable for chatbots, agents, simulations, and other stateful AI applications.

## Quick Start

Get up and running with Burr in minutes:

1.  **Install:** `pip install "burr[start]"`
2.  **Run the UI:** `burr`  (This opens the telemetry UI with sample data and a demo chatbot)
3.  **Explore Examples:**  Clone the repository and run a sample app:
    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    See the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for detailed instructions.

## How Burr Works

Burr simplifies building stateful applications by allowing you to define your workflow as a state machine. This provides clarity and control over complex processes, including:

*   **Simple Python Functions:** Define actions using Python functions.
*   **State Management:** Easily manage and track application state.
*   **Transitions:** Define the flow between actions using transitions.
*   **Extensive Integrations:** Integrate with LLMs, databases, and telemetry providers.

See the core API example in the original README.

## What Can You Build with Burr?

Burr is designed for a broad range of applications, including:

*   Chatbots (GPT-like, RAG-based)
*   LLM-powered games
*   Email assistants
*   Simulations
*   Hyperparameter tuning
*   And much more!

## Comparison with Other Frameworks

Burr's unique approach offers advantages compared to other popular frameworks:

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Roadmap

Burr is constantly evolving!  Future developments include:

*   FastAPI Integration and Hosted Deployment
*   Core library enhancements (retry, exception handling)
*   More integrations with popular frameworks (LCEL, LLamaIndex, Hamilton, etc...)
*   Tooling for hosted state machine execution
*   Additional storage integrations

Stay updated on the Burr Cloud waitlist: [Sign up here](https://forms.gle/w9u2QKcPrztApRedA)

## Contributing

Contributions are welcome!  See the [developer docs](https://burr.dagworks.io/contributing) to get started.

## Testimonials

Read what users are saying about Burr (as in the original README).

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