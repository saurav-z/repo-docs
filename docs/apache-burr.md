# Burr: Build Stateful AI Applications with Ease

<p align="center">
    <img src="https://github.com/user-attachments/assets/2ab9b499-7ca2-4ae9-af72-ccc775f30b4e" width=25 height=25/>
</p>

Burr is a Python framework that empowers you to build and manage state machines, enabling you to develop robust and scalable AI applications.

[Join the Burr Discord](https://discord.gg/6Zy2DwP4f3) | [Downloads](https://pepy.tech/project/burr) | [GitHub Last Commit](https://img.shields.io/github/last-commit/dagworks-inc/burr) | [Follow on X (formerly Twitter)](https://twitter.com/burr_framework) | [Follow DAGWorks on LinkedIn](https://linkedin.com/showcase/dagworks-inc) | [Follow DAGWorks on X (formerly Twitter)](https://twitter.com/dagworks)

## Key Features

*   **State Machine Foundation:** Express your applications as state machines, making complex decision-making processes clear and manageable.
*   **Simple Python Building Blocks:** Utilize straightforward Python functions to define actions and transitions, minimizing dependencies.
*   **Real-time UI for Monitoring & Debugging:** A built-in UI tracks, monitors, and traces your system's execution, aiding in introspection and debugging.
*   **Pluggable Persistence:** Easily save and load application state with pluggable persisters for data management.
*   **Framework Agnostic:** Burr integrates with your favorite frameworks and LLMs.
*   **Open-Source:** Built with the goal to build a powerful, easy-to-use, and open-source framework.

## Quick Start

Install Burr:

```bash
pip install "burr[start]"
```

Then, run the UI:

```bash
burr
```

The UI showcases a demo chat application.  Access the "Demos" sidebar and select `chatbot`. You'll need to set your `OPENAI_API_KEY` environment variable to fully experience the chat functionality.

Clone, run examples:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

For detailed installation and usage, see the [documentation](https://burr.dagworks.io/).

## How Burr Works

Burr allows you to design complex workflows by expressing your application logic as a state machine.  It's ideal for managing state, tracking decisions, incorporating human feedback, and creating idempotent, self-persisting workflows.

Here's a simplified example of how it works:

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

Burr offers:

1.  A lightweight, dependency-free Python library for building and managing state machines.
2.  A user-friendly UI for execution telemetry, introspection, and debugging.
3.  Integrations for state persistence, telemetry connection, and system integration.

<img src="https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif" alt="Burr at work" width="80%"/>

## Use Cases

Burr can be used in various applications, including:

1.  [Simple GPT-like chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/multi-modal-chatbot)
2.  [Stateful RAG-based chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/conversational-rag/simple_example)
3.  [LLM-based adventure game](https://github.com/DAGWorks-Inc/burr/tree/main/examples/llm-adventure-game)
4.  [Interactive email assistant](https://github.com/DAGWorks-Inc/burr/tree/main/examples/email-assistant)

## Comparison with Other Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why the Name?

Burr is named after Aaron Burr, a founding father. This project is the second open-source library release from DAGWorks, following the [Hamilton library](https://github.com/dagworks-inc/hamilton).

## Testimonials

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

*   FastAPI integration + hosted deployment
*   Core library efficiency/usability improvements
    *   First-class support for retries + exception management
    *   More integration with popular frameworks (LCEL, LLamaIndex, Hamilton, etc...)
    *   Capturing & surfacing extra metadata, e.g. annotations for particular point in time, that you can then pull out for fine-tuning, etc.
    *   Improvements to the pydantic-based typing system
*   Tooling for hosted execution of state machines, integrating with your infrastructure (Ray, modal, FastAPI + EC2, etc...)
*   Additional storage integrations. More integrations with technologies like MySQL, S3, etc. so you can run Burr on top of what you have available.

[Sign up for the Burr Cloud waitlist](https://forms.gle/w9u2QKcPrztApRedA)

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) to start.

## Contributors

### Code Contributions

-   [Elijah ben Izzy](https://github.com/elijahbenizzy)
-   [Stefan Krawczyk](https://github.com/skrawcz)
-   [Joseph Booth](https://github.com/jombooth)
-   [Nandani Thakur](https://github.com/NandaniThakur)
-   [Thierry Jean](https://github.com/zilto)
-   [Hamza Farhan](https://github.com/HamzaFarhan)
-   [Abdul Rafay](https://github.com/proftorch)
-   [Margaret Lange](https://github.com/margaretlange)

### Bug Hunters/Special Mentions

-   [Luke Chadwick](https://github.com/vertis)
-   [Evans](https://github.com/sudoevans)
-   [Sasmitha Manathunga](https://github.com/mmz-001)

[Back to the top](#burr-build-stateful-ai-applications-with-ease)