# Burr: Build Powerful, Stateful AI Applications with Ease

Burr is a Python library that simplifies building stateful AI applications like chatbots, agents, and simulations, offering a clean and efficient way to manage complex decision-making processes. [Explore the Burr repository](https://github.com/apache/burr).

**Key Features:**

*   **State Machine Abstraction:** Express your application logic as a state machine for clarity and control.
*   **Framework Agnostic:** Integrate with your favorite LLMs and frameworks seamlessly.
*   **Real-time UI:** Monitor, track, and debug your system's execution with Burr's intuitive UI.
*   **Pluggable Persisters:** Save and load application state using various storage options.
*   **Simple Python API:**  Build applications with straightforward Python functions.
*   **Versatile Applications:** Suitable for chatbots, RAG systems, LLM-powered games, email assistants, simulations, hyperparameter tuning, and more.

## Getting Started

1.  **Installation:**

    ```bash
    pip install "burr[start]"
    ```

2.  **Run the UI:**

    ```bash
    burr
    ```

    Access the telemetry UI to visualize your application's state.

3.  **Example:**

    ```bash
    git clone https://github.com/apache/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    See the counter example running in your terminal, and the trace being tracked in the UI.

    Refer to the [Getting Started Guide](https://burr.dagworks.io/getting_started/simple-example/) for detailed instructions.

## How Burr Works

Burr employs a simple yet powerful core API:

*   Define actions using Python functions.
*   Specify state transitions to create a state machine.
*   Run the application, tracking state changes in real-time.

## Benefits of Using Burr

*   **Improved Code Organization:** Structure complex AI applications with a clear state machine design.
*   **Simplified Debugging:** Utilize the UI to visualize and troubleshoot application behavior.
*   **Enhanced Scalability:**  Easily adapt to evolving application needs with its modular design.

## What You Can Build with Burr

*   GPT-like chatbots
*   Stateful RAG-based chatbots
*   LLM-based adventure games
*   Interactive email assistants
*   Time-series forecasting simulations
*   Hyperparameter tuning applications

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
>
> **Ashish Ghosh**, *CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
>
> **Reddit user cyan2k**, *LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
>
> **Ishita**, *Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
>
> **Matthew Rideout**, *Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
>
> **Rinat Gareev**, *Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
>
> **Hadi Nayebi**, *Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."
>
> **Aditya K.**, *DS Architect, TaskHuman*

## Roadmap

*   FastAPI integration + hosted deployment
*   Core library enhancements (retries, exception management, integration with other frameworks)
*   Tooling for hosted execution
*   Additional storage integrations

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) for details.

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