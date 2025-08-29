# Burr: Build and Manage Stateful AI Applications

**Burr is a Python library that simplifies the development of stateful AI applications, enabling you to build chatbots, agents, simulations, and more with ease.**

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

[View the original repository on GitHub](https://github.com/apache/burr)

## Key Features

*   **State Machine Modeling:** Express your application logic as a state machine for clear, maintainable code.
*   **UI for Telemetry:**  Visualize execution traces in real-time with Burr's built-in UI for easy debugging and monitoring.
*   **Framework-Agnostic:** Integrate Burr with your favorite LLMs and frameworks.
*   **Pluggable Persistence:** Save and load application state with pluggable persisters.
*   **Versatile Applications:** Build chatbots, agents, simulations, and more.
*   **Customizable Actions:** Easily integrate with your existing tools and libraries.

## Getting Started

1.  **Installation:** Install Burr using pip:

    ```bash
    pip install "burr[start]"
    ```
2.  **Run the UI:** Start the Burr telemetry UI:

    ```bash
    burr
    ```

    This opens the UI, which includes demo data and a chatbot example to help you get started.  The UI provides real-time monitoring of your application.
3.  **Run Examples:** Clone the repository and run the hello-world counter example:

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    Observe the trace in the UI.

4.  **Explore the Documentation:** For detailed information and more examples, see the [Burr documentation](https://burr.dagworks.io/).

## How Burr Works

Burr's core API allows you to create state machines using simple Python functions (actions) and transitions.  You can manage state, track decisions, and create idempotent workflows with ease.  

**Example Code Snippet:**

```python
from burr.core import action, State, ApplicationBuilder

@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    # your code -- write what you want here, for example
    chat_item = {"role" : "user", "content" : prompt}
    return state.update(prompt=prompt).append(chat_history=chat_item)

@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
    # query the LLM however you want (or don't use an LLM, up to you...)
    response = _query_llm(state["chat_history"]) # Burr doesn't care how you use LLMs!
    chat_item = {"role" : "system", "content" : response}
    return state.update(response=content).append(chat_history=chat_item)

app = (
    ApplicationBuilder()
    .with_actions(human_input, ai_response)
    .with_transitions(
        ("human_input", "ai_response"),
        ("ai_response", "human_input")
    ).with_state(chat_history=[])
    .with_entrypoint("human_input")
    .build()
)
*_, state = app.run(halt_after=["ai_response"], inputs={"prompt": "Who was Aaron Burr, sir?"})
print("answer:", app.state["response"])
```

Burr provides a low-abstraction Python library, a UI for introspection, and integrations to make it easy to connect to telemetry, and integrate with other systems.

## What Can You Build with Burr?

*   GPT-like chatbots
*   Stateful RAG-based chatbots
*   LLM-based adventure games
*   Interactive email assistants
*   Simulations
*   Hyperparameter tuning applications

And much more!  Burr lets you focus on your application logic while it handles the complexities of state management and workflow.

## Comparison with Other Frameworks

Burr offers a unique approach to managing state in AI applications:

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why the name "Burr"?

Named after Aaron Burr, Burr is the second open-source library from DAGWorks, following the [Hamilton library](https://github.com/dagworks-inc/hamilton).  It was originally developed to handle state between executions of Hamilton DAGs, and has expanded to a more general-purpose tool for building stateful AI applications.

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

Burr is constantly evolving! Planned features include:

1.  FastAPI integration + hosted deployment.
2.  Efficiency and usability improvements.
3.  Tooling for hosted execution.
4.  Additional storage integrations.
5.  Burr Cloud is coming soon!  Sign up for the waitlist [here](https://forms.gle/w9u2QKcPrztApRedA).

## Contributing

We welcome contributions! See the [developer documentation](https://burr.dagworks.io/contributing) to get started.

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