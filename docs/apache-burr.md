# Burr: Build, Monitor, and Scale Stateful AI Applications

[<img src="https://github.com/user-attachments/assets/2ab9b499-7ca2-4ae9-af72-ccc775f30b4e" width=25 height=25/>](https://github.com/apache/burr)

**Burr simplifies building and managing complex AI applications, offering a powerful state machine framework with real-time monitoring.**

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

*   **State Machine Framework:** Build applications using a clear, stateful approach with Python functions.
*   **Real-time UI:** Monitor, trace, and debug your application's execution in real-time.
*   **Pluggable Persisters:** Save and load application state using various storage integrations.
*   **Framework Agnostic:** Integrates with your favorite LLM and other frameworks.
*   **Extensible:** Customize with hooks and integrations, including Hamilton.
*   **Open Source:** Benefit from a community-driven project with comprehensive documentation.

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

### Run the UI

Run the UI server to view and monitor your application's execution.

```bash
burr
```

Explore the demo chatbot application in the UI to experience Burr's capabilities.
*   Demo chatbot requires the `OPENAI_API_KEY` environment variable set.

### Example

Clone the repository and run the hello-world counter example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

View the counter's trace in the UI.

## How Burr Works

Burr represents your application as a state machine, providing a simple API for managing state, tracking decisions, and creating idempotent workflows.

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

Burr's core components include:

1.  A dependency-free Python library for building and managing state machines.
2.  A UI for execution telemetry.
3.  Integrations for state persistence, telemetry, and system integration.

[![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)]

## Use Cases

Burr powers a variety of applications:

1.  GPT-like chatbot
2.  Stateful RAG-based chatbot
3.  LLM-based adventure game
4.  Interactive assistant for writing emails
5.  Time-series forecasting simulation
6.  Hyperparameter tuning

Integrate with your favorite LLM frameworks, observability tools, and storage solutions. Burr manages the workflow; you handle your model, data, and API integrations.

## Comparison

Compare Burr with similar frameworks:

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why Burr?

Burr is named after Aaron Burr, the nemesis of Alexander Hamilton. Like the Hamilton library, Burr provides a robust solution for managing state and workflows in your applications.

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

Future enhancements include:

1.  FastAPI integration + hosted deployment
2.  Efficiency/usability improvements
3.  Tooling for hosted execution
4.  Additional storage integrations

Sign up for the Burr Cloud waitlist [here](https://forms.gle/w9u2QKcPrztApRedA)

## Contribute

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) for more details.

## Contributors

### Code Contributions

Users who have contributed core functionality, integrations, or examples.

-   [Elijah ben Izzy](https://github.com/elijahbenizzy)
-   [Stefan Krawczyk](https://github.com/skrawcz)
-   [Joseph Booth](https://github.com/jombooth)
-   [Nandani Thakur](https://github.com/NandaniThakur)
-   [Thierry Jean](https://github.com/zilto)
-   [Hamza Farhan](https://github.com/HamzaFarhan)
-   [Abdul Rafay](https://github.com/proftorch)
-   [Margaret Lange](https://github.com/margaretlange)

### Bug hunters/special mentions

Users who have contributed small docs fixes, design suggestions, and found bugs

-   [Luke Chadwick](https://github.com/vertis)
-   [Evans](https://github.com/sudoevans)
-   [Sasmitha Manathunga](https://github.com/mmz-001)

[**Explore the Burr repository on GitHub**](https://github.com/apache/burr)