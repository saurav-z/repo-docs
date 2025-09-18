# Burr: Build and Manage Stateful AI Applications with Ease

Burr simplifies the development of decision-making applications like chatbots, agents, and simulations using Python.  [(View on GitHub)](https://github.com/apache/burr)

<div align="center">
  <a href="https://discord.gg/6Zy2DwP4f3">
    <img src="https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord" alt="Join Burr Discord">
  </a>
  <a href="https://pepy.tech/project/burr">
    <img src="https://static.pepy.tech/badge/burr/month" alt="Monthly Downloads">
  </a>
  <img src="https://static.pepy.tech/badge/burr" alt="PyPI Downloads">
  <a href="https://github.com/dagworks-inc/burr/pulse">
    <img src="https://img.shields.io/github/last-commit/dagworks-inc/burr" alt="Last Commit">
  </a>
  <a href="https://twitter.com/burr_framework" target="_blank">
    <img src="https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social" alt="Follow on X">
  </a>
  <a href="https://www.linkedin.com/showcase/dagworks-inc" target="_blank">
    <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin" alt="Follow on LinkedIn">
  </a>
  <a href="https://twitter.com/burr_framework" target="_blank">
    <img src="https://img.shields.io/badge/burr_framework-Follow-purple.svg?logo=X" alt="Follow Burr on X">
  </a>
  <a href="https://twitter.com/dagworks" target="_blank">
    <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=X" alt="Follow DAGWorks on X">
  </a>
</div>

Burr provides a powerful framework for building stateful AI applications, with a focus on ease of use, real-time monitoring, and flexible integration with your favorite tools.

## Key Features

*   **State Machine Design:** Express your application logic as state machines for clear, maintainable code.
*   **Real-time Monitoring:** Built-in UI for tracing, monitoring, and debugging application execution.
*   **Flexible Integrations:** Pluggable persisters and integrations to connect with LLMs, observability tools, storage solutions, and more.
*   **Framework-Agnostic:** Works seamlessly with various frameworks, allowing you to leverage existing tools and workflows.
*   **Python-Based:** Built using simple python building blocks.
*   **Open Source:** Free and open source under the Apache License 2.0.

## Quick Start

1.  **Install:**

    ```bash
    pip install "burr[start]"
    ```

2.  **Run the UI:**

    ```bash
    burr
    ```

    This opens the Burr telemetry UI, which includes a demo chatbot. Access the "Demos" sidebar and select "chatbot".  You will need an OPENAI_API_KEY to chat.

3.  **Run a simple example:**

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```

    Check the UI to see the example execution.

    For detailed guidance, refer to the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr lets you build applications as state machines.  Define actions with simple python functions.

**Example:**

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

Burr includes:

1.  A Python library for building and managing state machines.
2.  A UI to visualize execution telemetry.
3.  Integrations for persistence, telemetry, and system integrations.

  <p align="center">
  <img src="https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif" alt="Burr in Action" width="600">
  </p>

## Use Cases

Burr supports a variety of AI applications, including:

1.  [Simple GPT-like chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/multi-modal-chatbot)
2.  [Stateful RAG-based chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/conversational-rag/simple_example)
3.  [LLM-based adventure game](https://github.com/DAGWorks-Inc/burr/tree/main/examples/llm-adventure-game)
4.  [Interactive assistant for writing emails](https://github.com/DAGWorks-Inc/burr/tree/main/examples/email-assistant)

Also, Burr can be used in simulations and hyperparameter tuning.

## Build Your Application

See the documentation for [getting started](https://burr.dagworks.io/getting_started/simple-example) to get started.

## Comparison with Other Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why the name "Burr"?

Named after Aaron Burr, the framework's purpose mirrors the dynamic of balancing distinct components in a state machine.

## Testimonials

(Added testimonials for social proof and SEO)

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."
>
> **Ashish Ghosh** *CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."
>
> **Reddit user cyan2k** *LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."
>
> **Ishita** *Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."
>
> **Matthew Rideout** *Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."
>
> **Rinat Gareev** *Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."
>
> **Hadi Nayebi** *Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."
>
> **Aditya K.** *DS Architect, TaskHuman*

## Roadmap

Future developments include:

1.  FastAPI integration + hosted deployment.
2.  Efficiency and usability improvements to the core library.
    *   First-class support for retries + exception management
    *   More integration with popular frameworks (LCEL, LLamaIndex, Hamilton, etc...)
    *   Capturing & surfacing extra metadata.
    *   Improvements to the pydantic-based typing system
3.  Tooling for hosted execution of state machines.
4.  Additional storage integrations.

Sign up for the [Burr Cloud waitlist](https://forms.gle/w9u2QKcPrztApRedA).

## Contribute

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

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