# Burr: Build and Manage Stateful AI Applications with Ease

**Burr is a Python library that simplifies the development of stateful AI applications, providing a streamlined approach to building chatbots, agents, and more.**  Find the source code and contribute at [https://github.com/apache/burr](https://github.com/apache/burr).

<div align="center">
  <a href="https://discord.gg/6Zy2DwP4f3">
    <img src="https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord" alt="Join Burr Discord">
  </a>
  <a href="https://pepy.tech/project/burr">
    <img src="https://static.pepy.tech/badge/burr/month" alt="PyPI Downloads">
  </a>
  <img src="https://static.pepy.tech/badge/burr" alt="PyPI Downloads">
  <a href="https://github.com/dagworks-inc/burr/pulse">
    <img src="https://img.shields.io/github/last-commit/dagworks-inc/burr" alt="GitHub Last Commit">
  </a>
  <a href="https://twitter.com/burr_framework">
    <img src="https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social" alt="Follow @burr_framework on X">
  </a>
  <a href="https://www.linkedin.com/showcase/dagworks-inc" style="background:none">
    <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin" alt="Follow DAGWorks on LinkedIn" />
  </a>
  <a href="https://twitter.com/burr_framework" target="_blank">
    <img src="https://img.shields.io/badge/burr_framework-Follow-purple.svg?logo=X" alt="Follow Burr Framework on X"/>
  </a>
  <a href="https://twitter.com/dagworks" target="_blank">
    <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=X" alt="Follow DAGWorks on X"/>
  </a>
</div>

## Key Features:

*   **State Machine Design:** Express your application as a state machine for clear, manageable workflows.
*   **Real-time UI:** Monitor, trace, and debug your AI applications with a built-in user interface.
*   **Framework Agnostic:** Integrate with your favorite LLMs and frameworks seamlessly.
*   **Pluggable Persistence:** Easily save and load application state with various storage options.
*   **Versatile Applications:** Ideal for chatbots, agents, simulations, and other stateful AI systems.

## Getting Started:

1.  **Installation:** Install Burr using pip:

    ```bash
    pip install "burr[start]"
    ```

    (See the [documentation](https://burr.dagworks.io/getting_started/install/) if you are using poetry.)

2.  **Run the UI:** Start the Burr UI server:

    ```bash
    burr
    ```

    This opens the Burr telemetry UI. Explore the demo chatbot application (requires `OPENAI_API_KEY`).
3.  **Example:** Run the hello-world counter example:

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    Observe the counter in the terminal and trace in the UI. For more details, check out the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works:

Burr simplifies state management using a core API centered around state machines. You define actions and transitions to create and manage your application's logic.

```python
from burr.core import action, State, ApplicationBuilder

@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    chat_item = {"role" : "user", "content" : prompt}
    return state.update(prompt=prompt).append(chat_history=chat_item)

@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
    response = _query_llm(state["chat_history"])
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

*   **Core Library:** A lightweight, dependency-free Python library for building and managing state machines.
*   **UI:** A user interface for introspection and debugging via execution telemetry.
*   **Integrations:** Tools for state persistence, telemetry connection, and system integration.

![Burr in Action](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## Use Cases:

Burr empowers developers to build a wide range of AI applications:

1.  [GPT-like Chatbots](https://github.com/dagworks-inc/burr/tree/main/examples/multi-modal-chatbot)
2.  [Stateful RAG-based Chatbots](https://github.com/dagworks-inc/burr/tree/main/examples/conversational-rag/simple_example)
3.  [LLM-based Adventure Games](https://github.com/DAGWorks-Inc/burr/tree/main/examples/llm-adventure-game)
4.  [Interactive Email Assistants](https://github.com/DAGWorks-Inc/burr/tree/main/examples/email-assistant)
5.  Simulations
6.  Hyperparameter tuning

Use hooks and other integrations to seamlessly integrate with your preferred vendors (LLM observability, storage, etc.) and build custom actions that delegate to your favourite libraries (such as [Hamilton](https://github.com/DAGWorks-Inc/hamilton)). Burr will help you structure your application in a way that scales with your needs and simplifies following your system's logic.

## Comparison:

| Feature                                      | Burr | Langgraph | Temporal | Langchain | Superagent | Hamilton |
| -------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicit State Machine Modeling              |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-Agnostic                           |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous Event-Based Orchestration      |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Web-Service Logic                            |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Monitoring/Tracing UI                        |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Non-LLM Use Cases                           |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why Burr?

Burr is named after Aaron Burr, inspired by the Hamilton project, reflecting the potential for different approaches to work together for a common good. Burr was designed to handle state management between Hamilton DAG executions.

## Testimonials:

*   **Ashish Ghosh** (CTO, Peanut Robotics): "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
*   **Reddit user cyan2k** (LocalLlama, Subreddit): "Honestly, take a look at Burr. Thank me later."
*   **Ishita** (Founder, Watto.ai): "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
*   **Matthew Rideout** (Staff Software Engineer, Paxton AI): "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
*   **Rinat Gareev** (Senior Solutions Architect, Provectus): "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
*   **Hadi Nayebi** (Co-founder, CognitiveGraphs): "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
*   **Aditya K.** (DS Architect, TaskHuman): "Moving from LangChain to Burr was a game-changer! It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain. With Burr, I could finally have a cleaner, more sophisticated, and stable implementation."

## Roadmap:

*   FastAPI Integration + Hosted Deployment
*   Efficiency/Usability Improvements
    *   Retries + Exception Management
    *   More integrations with LCEL, LlamaIndex, and Hamilton.
    *   Metadata capture and surfacing for fine-tuning.
    *   Improvements to the pydantic-based typing system.
*   Hosted Execution Tooling
*   Additional Storage Integrations
    *   Sign up for Burr Cloud waitlist: [here](https://forms.gle/w9u2QKcPrztApRedA)

## Contributing:

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Contributors:

A list of the code contributors and bug hunters/special mentions can be found in the original README.