# Burr: Build and Monitor Stateful AI Applications with Ease

Burr empowers developers to build, monitor, and debug stateful AI applications using simple Python building blocks. [Explore the original repo](https://github.com/apache/burr).

---

## Key Features

*   **State Machine Foundation:** Express your application logic as a state machine for clarity and control.
*   **UI Telemetry:** Real-time monitoring, tracing, and debugging through a built-in UI.
*   **Framework Agnostic:** Integrate Burr with your favorite LLM frameworks and libraries.
*   **Pluggable Persisters:** Save and load application state with ease using various storage options.
*   **Versatile Use Cases:** Supports chatbots, agents, simulations, and more.

---

## Quick Start

**Install Burr:**

```bash
pip install "burr[start]"
```

**Run the UI:**

```bash
burr
```

*   Access the telemetry UI and explore example data.
*   Test the demo chatbot (requires `OPENAI_API_KEY`).
*   Follow the [documentation](https://burr.dagworks.io/) and [getting started guide](https://burr.dagworks.io/getting_started/simple-example/)

**Run a sample application:**

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

---

## How Burr Works

Burr uses a straightforward API to define actions and transitions within a state machine. Build robust applications with the following key elements:

*   **Actions:** Simple Python functions that read from and write to the application state.
*   **Transitions:** Define the flow between actions, forming your application's logic.
*   **State:** Manage the data that your application operates on.

Burr offers flexibility and control, including:

*   A lightweight Python library with no external dependencies.
*   A UI for inspecting your application's execution in real-time.
*   Integrations for state persistence, telemetry, and vendor compatibility.

---

## What You Can Build

Burr is ideal for a wide range of applications:

*   Chatbots (GPT-like, RAG-based, etc.)
*   LLM-powered adventure games
*   Email assistants
*   Simulations and hyperparameter tuning

Burr simplifies the process of constructing scalable and maintainable AI applications by decoupling your business logic and your state management.

---

## Comparison with other Frameworks

Burr offers a unique approach to building stateful AI applications:

| Criteria                                          | Burr | Langgraph | Temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

---

## Testimonials

Read what users are saying about Burr:

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making." - *Ashish Ghosh, CTO, Peanut Robotics*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top." - *Ishita, Founder, Watto.ai*

---

## Roadmap

Burr continues to evolve with new features:

*   FastAPI integration and hosted deployment
*   Improvements to core library
*   Tooling for hosted state machine execution
*   Additional storage integrations

---

## Contribute

Join the Burr community! See the [developer-facing docs](https://burr.dagworks.io/contributing) for contributing.