# Burr: Build and Monitor Stateful AI Applications

**Burr simplifies the development of AI applications by providing a robust framework for managing state, tracing execution, and integrating with your favorite tools.**  [Visit the Burr GitHub Repository](https://github.com/apache/burr)

<div align="center">
  <a href="https://discord.gg/6Zy2DwP4f3"><img src="https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord" alt="Discord"></a>
  <a href="https://pepy.tech/project/burr"><img src="https://static.pepy.tech/badge/burr/month" alt="Downloads"></a>
  <img src="https://static.pepy.tech/badge/burr" alt="PyPI Downloads">
  <a href="https://github.com/dagworks-inc/burr/pulse"><img src="https://img.shields.io/github/last-commit/dagworks-inc/burr" alt="Last Commit"></a>
  <a href="https://twitter.com/burr_framework" target="_blank"><img src="https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social" alt="X"></a>
  <a href="https://linkedin.com/showcase/dagworks-inc" target="_blank"><img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin" alt="LinkedIn"></a>
  <a href="https://twitter.com/burr_framework" target="_blank"><img src="https://img.shields.io/badge/burr_framework-Follow-purple.svg?logo=X" alt="X"></a>
  <a href="https://twitter.com/dagworks" target="_blank"><img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=X" alt="X"></a>
</div>

## Key Features

*   **State Machine Foundation:** Define your application logic as a state machine, enabling clear management of complex workflows.
*   **Real-time Monitoring:**  Built-in UI for execution telemetry, allowing for easy introspection and debugging.
*   **Framework-Agnostic:**  Integrates seamlessly with your preferred LLM providers, frameworks, and libraries.
*   **Pluggable Persisters:**  Save and load application state with a variety of storage options.
*   **Open Source & Community Driven:**  Active community with open source contributions.

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

(See the [documentation](https://burr.dagworks.io/getting_started/install/) for more installation options)

### Running the UI

Start the Burr UI to visualize your state machine's execution:

```bash
burr
```

The UI includes demo data, a chatbot example (requires an OpenAI API key), and provides real-time insights into your application.

### Example

Clone the repository and run a simple counter example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

This will run a counter example and show the execution trace in the UI.

For more details, see the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr allows you to build applications as state machines, offering a powerful way to manage state, track decisions, and create idempotent workflows.  The core API is simple:

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

Burr provides:

1.  A Python library to build and manage state machines.
2.  A UI for execution telemetry and debugging.
3.  Integrations for state persistence, telemetry, and third-party system connections.

<img src="https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif" alt="Burr at work" width="70%">

## Use Cases

Burr is ideal for applications requiring state management and workflow orchestration, including:

*   Chatbots (GPT-like, RAG-based)
*   LLM-based adventure games
*   Email assistants
*   Simulations
*   Hyperparameter tuning

##  Integrations

Burr integrates with various tools and vendors, and allows you to:

*   Connect to your favorite LLM providers.
*   Utilize storage solutions.
*   Build custom actions.

## Comparison with Other Frameworks

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

*   FastAPI integration and deployment.
*   Improvements to the core library (retries, exception management, framework integrations).
*   Hosted execution and infrastructure integrations.
*   Additional storage integrations.
*   Burr Cloud.

## Contributing

Contributions are welcome!  See the [developer documentation](https://burr.dagworks.io/contributing) to get started.

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