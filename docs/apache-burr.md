# Burr: Build and Manage Stateful AI Applications

**Burr simplifies the development of stateful AI applications using a Python-based state machine framework and real-time UI, allowing you to easily build chatbots, agents, simulations, and more.**

[<img src="https://github.com/user-attachments/assets/2ab9b499-7ca2-4ae9-af72-ccc775f30b4e" width=25 height=25/>  Burr on GitHub](https://github.com/apache/burr)

[![Discord](https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord)](https://discord.gg/6Zy2DwP4f3)
[![Downloads](https://static.pepy.tech/badge/burr/month)](https://pepy.tech/project/burr)
[![PyPI Downloads](https://static.pepy.tech/badge/burr)](https://pepy.tech/project/burr)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/dagworks-inc/burr)](https://github.com/dagworks-inc/burr/pulse)
[![X](https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social)](https://twitter.com/burr_framework)
[![LinkedIn](https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin)](https://linkedin.com/showcase/dagworks-inc)
[![X](https://img.shields.io/badge/burr_framework-Follow-purple.svg?logo=X)](https://twitter.com/burr_framework)
[![X](https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=X)](https://twitter.com/dagworks)

**Key Features:**

*   **Python-Based State Machines:** Define your application logic with simple Python functions and state transitions.
*   **Real-time UI:** Monitor, trace, and debug your applications with an interactive UI.
*   **Flexible Integrations:** Easily integrate with your favorite LLMs, frameworks, and storage solutions.
*   **Idempotent Workflows:** Build applications that can handle state management, complex decisions, and human feedback.
*   **Open-Source:** Leverage a powerful and adaptable framework for building AI applications.

**Getting Started**

1.  **Installation:** `pip install "burr[start]"`
2.  **Run UI:** `burr`  (access the telemetry UI)
3.  **Explore Examples:**  Clone the repository and run the hello-world counter example: `git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter && python application.py`

For detailed instructions, see the [documentation](https://burr.dagworks.io/) and the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).  Also, check out the quick video intro [here](https://www.loom.com/share/a10f163428b942fea55db1a84b1140d8?sid=1512863b-f533-4a42-a2f3-95b13deb07c9) and longer intro with walkthrough [here](https://www.youtube.com/watch?v=rEZ4oDN0GdU).

**How Burr Works**

Burr uses a simple core API for creating state machines. The core API is simple -- the Burr hello-world looks like this:

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

**What You Can Build with Burr**

Burr empowers you to build:

1.  Simple GPT-like chatbots
2.  Stateful RAG-based chatbots
3.  LLM-based adventure games
4.  Interactive email assistants
5.  Simulations (e.g., time-series forecasting)
6.  Hyperparameter tuning

**Comparison Against Other Frameworks**

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

**Why the Name Burr?**

Burr is named after Aaron Burr, and is the second open-source library release after the Hamilton library.

**Testimonials**

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

**Roadmap**

*   FastAPI integration + hosted deployment
*   Efficiency/usability improvements
*   Tooling for hosted execution
*   Additional storage integrations
*   Burr Cloud - Sign up [here](https://forms.gle/w9u2QKcPrztApRedA)

**Contributing**

We welcome contributions!  See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

**Contributors**

*   [Elijah ben Izzy](https://github.com/elijahbenizzy)
*   [Stefan Krawczyk](https://github.com/skrawcz)
*   [Joseph Booth](https://github.com/jombooth)
*   [Nandani Thakur](https://github.com/NandaniThakur)
*   [Thierry Jean](https://github.com/zilto)
*   [Hamza Farhan](https://github.com/HamzaFarhan)
*   [Abdul Rafay](https://github.com/proftorch)
*   [Margaret Lange](https://github.com/margaretlange)

**Bug Hunters/Special Mentions**

*   [Luke Chadwick](https://github.com/vertis)
*   [Evans](https://github.com/sudoevans)
*   [Sasmitha Manathunga](https://github.com/mmz-001)