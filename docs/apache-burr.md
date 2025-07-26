# Burr: Build and Monitor Stateful AI Applications

Burr empowers developers to easily create stateful AI applications with a focus on state management, monitoring, and debugging. [Explore the Burr repository](https://github.com/apache/burr).

*   **Stateful AI Made Easy:** Burr simplifies the development of applications like chatbots, agents, and simulations by allowing you to express your application as a state machine using simple Python building blocks.
*   **Real-time Monitoring and Debugging:** Includes a user-friendly UI to track, monitor, and trace your system's execution in real-time.
*   **Flexible Integrations:** Integrates seamlessly with your favorite frameworks and offers pluggable persisters for saving and loading application state.
*   **Wide range of use cases:** Chatbots, RAG based chatbots, LLM-based adventure games, and interactive assistants.

## Key Features

*   **State Machine Foundation:** Core API enables building and managing state machines using simple Python functions.
*   **Telemetry UI:** A UI for introspection and debugging, visualize execution telemetry.
*   **Extensive Integrations:** Easy to persist state, connect to telemetry, and integrate with other systems.

## Quick Start

1.  **Install:**
    ```bash
    pip install "burr[start]"
    ```
    (See [the docs](https://burr.dagworks.io/getting_started/install/) if you're using Poetry)
2.  **Run the UI:**
    ```bash
    burr
    ```
    This will open the Burr UI, including a demo chatbot (requires OPENAI\_API\_KEY).
3.  **Run an example:**
    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```
    See the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for more details.

## How Burr Works

Burr allows you to manage state, track decisions, add human feedback, and define idempotent workflows by expressing your application as a state machine. The core API offers a streamlined approach.

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

## Use Cases

Burr is versatile and can be applied to various applications:

*   [Simple GPT-like chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/multi-modal-chatbot)
*   [Stateful RAG-based chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/conversational-rag/simple_example)
*   [LLM-based adventure game](https://github.com/DAGWorks-Inc/burr/tree/main/examples/llm-adventure-game)
*   [Interactive assistant for writing emails](https://github.com/DAGWorks-Inc/burr/tree/main/examples/email-assistant)
*   Time-series forecasting [simulation](https://github.com/DAGWorks-Inc/burr/tree/main/examples/simulation)
*   [Hyperparameter tuning](https://github.com/DAGWorks-Inc/burr/tree/main/examples/ml-training)

## Comparison with Similar Frameworks

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

Upcoming features include:

1.  FastAPI integration + hosted deployment.
2.  Efficiency/usability improvements for the core library (e.g., retries, exception management, more framework integrations).
3.  Tooling for hosted execution.
4.  Additional storage integrations.

Sign up for the [Burr Cloud waitlist](https://forms.gle/w9u2QKcPrztApRedA) for the waitlist to get access.

## Contributing

We welcome contributors! See the [developer-facing docs](https://burr.dagworks.io/contributing) to start contributing.

## Contributors

### Code contributions

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