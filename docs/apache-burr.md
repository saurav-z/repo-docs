# Burr: Build and Monitor Stateful AI Applications

**Burr empowers you to easily develop and manage stateful AI applications, such as chatbots, agents, and simulations, using Python building blocks.**

[Go to the original repository](https://github.com/apache/burr)

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

*   **Simplified State Management:** Build and manage state machines for complex decision-making processes.
*   **Real-time Monitoring & Debugging:**  Built-in UI for tracking, monitoring, and tracing application execution.
*   **Flexible Integration:** Seamlessly integrates with popular LLM frameworks and other tools.
*   **Pluggable Persistence:** Easily save and load application state with pluggable persisters.
*   **Framework Agnostic:** Burr is designed to be flexible and can be used with any of your favorite frameworks.

## Quick Start

1.  **Install:** `pip install "burr[start]"`
2.  **Run UI:** `burr`
3.  **Explore:** Open the UI in your browser (usually http://localhost:8000/) and explore the included demo chatbot.
4.  **Run Example:**
    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```
    Then view the state machine execution in the UI.

    See the [documentation](https://burr.dagworks.io/getting_started/install/) if you're using poetry.

    For more details see the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr allows you to express your application logic as a state machine, making it easy to manage state, track decisions, and create self-persisting workflows. Here's a simple example:

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

Burr's key components include:

1.  **Core Library:** A lightweight, dependency-free Python library for building and managing state machines.
2.  **UI:** A user interface for visualizing execution telemetry, aiding in debugging and introspection.
3.  **Integrations:**  Tools for persisting state, connecting to telemetry, and integrating with other systems.

[View the Burr at work demo](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## Use Cases

Burr is ideal for a wide range of applications, including:

1.  [Chatbots](https://github.com/dagworks-inc/burr/tree/main/examples/multi-modal-chatbot)
2.  [Stateful RAG-based chatbots](https://github.com/dagworks-inc/burr/tree/main/examples/conversational-rag/simple_example)
3.  [LLM-based adventure games](https://github.com/DAGWorks-Inc/burr/tree/main/examples/llm-adventure-game)
4.  [Interactive assistants for writing emails](https://github.com/dagworks-inc/burr/tree/main/examples/email-assistant)
5.  And various non-LLM applications like simulations and hyperparameter tuning.

## Comparison with Other Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

##  Why the Name "Burr"?

Burr is named after Aaron Burr, and its connection to Hamilton is due to being DAGWorks' second open-source library after the [Hamilton library](https://github.com/dagworks-inc/hamilton). The original Burr was built as a harness to handle state between executions of Hamilton DAGs.

## Testimonials

> "After evaluating several other obfuscating LLM frameworks, their elegant yet comprehensive state management solution proved to be the powerful answer to rolling out robots driven by AI decision-making."

> **Ashish Ghosh**
>*CTO, Peanut Robotics*

> "Of course, you can use it [LangChain], but whether it's really production-ready and improves the time from 'code-to-prod' [...], we've been doing LLM apps for two years, and the answer is no [...] All these 'all-in-one' libs suffer from this [...]. Honestly, take a look at Burr. Thank me later."

> **Reddit user cyan2k**
>*LocalLlama, Subreddit*

> "Using Burr is a no-brainer if you want to build a modular AI application. It is so easy to build with, and I especially love their UI which makes debugging a piece of cake. And the always-ready-to-help team is the cherry on top."

> **Ishita**
>*Founder, Watto.ai*

> "I just came across Burr and I'm like WOW, this seems like you guys predicted this exact need when building this. No weird esoteric concepts just because it's AI."

> **Matthew Rideout**
>*Staff Software Engineer, Paxton AI*

> "Burr's state management part is really helpful for creating state snapshots and building debugging, replaying, and even evaluation cases around that."

> **Rinat Gareev**
>*Senior Solutions Architect, Provectus*

> "I have been using Burr over the past few months, and compared to many agentic LLM platforms out there (e.g. LangChain, CrewAi, AutoGen, Agency Swarm, etc), Burr provides a more robust framework for designing complex behaviors."

> **Hadi Nayebi**
>*Co-founder, CognitiveGraphs*

> "Moving from LangChain to Burr was a game-changer!
> - **Time-Saving**: It took me just a few hours to get started with Burr, compared to the days and weeks I spent trying to navigate LangChain.
> - **Cleaner Implementation**: With Burr, I could finally have a cleaner, more sophisticated, and stable implementation. No more wrestling with complex codebases.
> - **Team Adoption**: I pitched Burr to my teammates, and we pivoted our entire codebase to it. It's been a smooth ride ever since."

> **Aditya K.**
>*DS Architect, TaskHuman*

## Roadmap

The team is planning to make these improvements:
1.  FastAPI integration + hosted deployment
2.  Various efficiency/usability improvements for the core library
    1.  First-class support for retries + exception management
    2.  More integration with popular frameworks (LCEL, LLamaIndex, Hamilton, etc...)
    3.  Capturing & surfacing extra metadata, e.g. annotations for particular point in time, that you can then pull out for fine-tuning, etc.
    4.  Improvements to the pydantic-based typing system
3.  Tooling for hosted execution of state machines, integrating with your infrastructure (Ray, modal, FastAPI + EC2, etc...)
4.  Additional storage integrations. More integrations with technologies like MySQL, S3, etc. so you can run Burr on top of what you have available.

Sign up [here](https://forms.gle/w9u2QKcPrztApRedA) for Burr Cloud waitlist.

## Contributing

Contributions are welcome! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Contributors

### Code contributions

- [Elijah ben Izzy](https://github.com/elijahbenizzy)
- [Stefan Krawczyk](https://github.com/skrawcz)
- [Joseph Booth](https://github.com/jombooth)
- [Nandani Thakur](https://github.com/NandaniThakur)
- [Thierry Jean](https://github.com/zilto)
- [Hamza Farhan](https://github.com/HamzaFarhan)
- [Abdul Rafay](https://github.com/proftorch)
- [Margaret Lange](https://github.com/margaretlange)

### Bug hunters/special mentions

- [Luke Chadwick](https://github.com/vertis)
- [Evans](https://github.com/sudoevans)
- [Sasmitha Manathunga](https://github.com/mmz-001)