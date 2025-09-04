# Burr: Build State-Aware AI Applications with Ease

Burr empowers developers to create robust and manageable AI-powered applications by providing a simple yet powerful framework for building state machines in Python.

[View the original repo on GitHub](https://github.com/apache/burr)

<br>

## Key Features:

*   **State Machine Foundation**: Express application logic as a state machine for clear, organized workflows.
*   **UI for Telemetry**: Monitor, trace, and debug your application's execution in real-time with a built-in UI.
*   **Flexible Integrations**: Connect with your favorite LLMs, storage solutions, and telemetry providers.
*   **Python-First Development**: Build and manage state machines with simple Python functions.
*   **Versatile Applications**: Suitable for chatbots, agents, simulations, and more.

<br>

## Getting Started:

1.  **Installation:**

    ```bash
    pip install "burr[start]"
    ```

    (See the [documentation](https://burr.dagworks.io/getting_started/install/) for alternative install methods.)
2.  **Run the UI:**

    ```bash
    burr
    ```

    This opens Burr's telemetry UI, including a demo chatbot.
3.  **Explore Examples:**

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    The counter example runs in the terminal and is tracked in the UI. See [the docs](https://burr.dagworks.io/getting_started/simple-example/) for more details.

<br>

## How Burr Works:

Burr utilizes a simple API where you represent your application as a state machine. The core concepts involve:

*   **Actions:** Python functions that perform specific tasks within the state machine.
*   **State:** Data that persists across actions, representing the application's current context.
*   **Transitions:** Define the flow and execution order of actions.

Burr includes a UI, helping with introspection and debugging, and provides integrations for:
*   Persisting State.
*   Connecting to Telemetry.
*   Integrating with other systems.

![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

<br>

## Use Cases:

Burr can power various AI applications, including:

*   GPT-like chatbots
*   Stateful RAG-based chatbots
*   LLM-based adventure games
*   Interactive email assistants
*   Simulations & Hyperparameter Tuning

<br>

## Comparison with Other Frameworks:

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

<br>

## Testimonials:

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
>
> **Ashish Ghosh**
> *CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
>
> **Reddit user cyan2k**
> *LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
>
> **Ishita**
> *Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
>
> **Matthew Rideout**
> *Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."

> **Rinat Gareev**
> *Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
>
> **Hadi Nayebi**
> *Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."
>
> **Aditya K.**
> *DS Architect, TaskHuman*

<br>

## Roadmap:

Burr continues to evolve with planned features, including:

1.  FastAPI integration.
2.  Core library improvements (e.g., retries, exception handling, integration with popular frameworks).
3.  Tooling for hosted state machine execution.
4.  Additional storage integrations.

Sign up [here](https://forms.gle/w9u2QKcPrztApRedA) to join the waitlist for Burr Cloud.

<br>

## Contributing:

Contributions are welcome! See the [developer-facing docs](https://burr.dagworks.io/contributing) to start developing.