# Burr: Build and Monitor Stateful AI Applications

[Burr](https://github.com/apache/burr) empowers developers to build and manage stateful AI applications with ease, providing real-time monitoring and flexible integrations.

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

## Key Features:

*   **Simple State Machine Abstraction:** Define your applications as state machines using Python functions.
*   **Real-time Monitoring UI:** Track, monitor, and debug your system's execution in a user-friendly interface.
*   **Pluggable Persisters:** Save and load application state with various storage options.
*   **Flexible Integrations:** Connect to your favorite LLMs, frameworks, and observability tools.
*   **Versatile Use Cases:** Build chatbots, agents, simulations, and more.

## Getting Started

### Installation

Install Burr using pip:

```bash
pip install "burr[start]"
```

See the [documentation](https://burr.dagworks.io/getting_started/install/) for poetry instructions.

### Run the UI

Start the Burr UI server:

```bash
burr
```

The UI will open, pre-loaded with demo data.  Use the demo chatbot (requires an OpenAI API key) to see real-time tracking.

### Example

Clone the repository and run a simple example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

Find the counter example in the UI.

For detailed instructions, refer to the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr allows you to build complex applications around state.  Core concepts:

*   Define actions with `@action` decorator
*   Specify reads and writes
*   Build application flows

Burr includes:

1.  A Python library to build and manage state machines.
2.  A UI for introspection and debugging.
3.  Integrations for persistence, telemetry, and system integration.

## Use Cases

Burr is suitable for various AI applications:

*   Chatbots
*   Stateful RAG Chatbots
*   LLM-based Adventure Games
*   Email Assistants
*   Time-series Forecasting Simulations
*   Hyperparameter Tuning

## Comparison

| Feature                                      | Burr | Langgraph | Temporal | Langchain | Superagent | Hamilton |
| -------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicit State Machine Modeling               |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-Agnostic                           |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous Event-Based Orchestration        |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Core Web-Service Logic                       |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-Source UI for Monitoring and Tracing   |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with Non-LLM Use Cases                 |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Roadmap

*   FastAPI Integration and Hosted Deployment
*   Core Library Improvements
*   Hosted Execution Tooling
*   Additional Storage Integrations

## Contributing

Contributions are welcome!  See the [developer docs](https://burr.dagworks.io/contributing) to get started.

## Testimonials

> "Elegant yet comprehensive state management solution..." - *Ashish Ghosh, CTO, Peanut Robotics*

> "Take a look at Burr. Thank me later." - *Reddit user cyan2k, LocalLlama Subreddit*

> "No-brainer if you want to build a modular AI application..." - *Ishita, Founder, Watto.ai*

> "Burr's state management part is really helpful..." - *Rinat Gareev, Senior Solutions Architect, Provectus*

> "Moving from LangChain to Burr was a game-changer!" - *Aditya K., DS Architect, TaskHuman*

## Contributors

*   [Elijah ben Izzy](https://github.com/elijahbenizzy)
*   [Stefan Krawczyk](https://github.com/skrawcz)
*   [Joseph Booth](https://github.com/jombooth)
*   and many more!

**[Visit the original repository for more details](https://github.com/apache/burr)**