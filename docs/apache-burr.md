# Burr: Build Stateful AI Applications with Ease

**Burr simplifies the development of stateful AI applications like chatbots, agents, and simulations using Python.**

[![Discord](https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord)](https://discord.gg/6Zy2DwP4f3)
[![Downloads](https://static.pepy.tech/badge/burr/month)](https://pepy.tech/project/burr)
[![PyPI Downloads](https://static.pepy.tech/badge/burr)]
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

[Visit the Burr Repository on GitHub](https://github.com/apache/burr)

## Key Features

*   **State Machine Modeling:** Express your application logic as a state machine for clear control and management of state.
*   **Intuitive Python API:**  Build and manage state machines with simple Python functions.
*   **Real-time UI:**  Visualize execution telemetry for debugging, monitoring, and tracing your AI application.
*   **Pluggable Persistence:** Save and load application state with various persisters (e.g., for memory).
*   **Framework Agnostic:** Burr integrates with your favorite LLMs, frameworks, and tools.

## Getting Started

### Installation

Install Burr using pip:

```bash
pip install "burr[start]"
```

### Run the UI

Start the Burr UI server:

```bash
burr
```

Explore the UI with pre-loaded demo data and a chatbot example (requires an OpenAI API key).

### Example:  Hello World Counter

Clone the repository and run the counter example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

This example showcases the tracking of state in the UI.

###  Documentation
For detailed instructions and examples, see the [Burr Documentation](https://burr.dagworks.io/).  Also, check out these resources:
*   Quick (<3min) video intro [here](https://www.loom.com/share/a10f163428b942fea55db1a84b1140d8?sid=1512863b-f533-4a42-a2f3-95b13deb07c9).
*   Longer [video intro & walkthrough](https://www.youtube.com/watch?v=rEZ4oDN0GdU).
*   Blog post [here](https://blog.dagworks.io/p/burr-develop-stateful-ai-applications).
*   Join the discord for support [here](https://discord.gg/6Zy2DwP4f3).

## How Burr Works

Burr lets you build applications as state machines (graphs/flowcharts). Here's a simplified example:

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

Burr offers:

1.  A low-abstraction, dependency-free Python library for building and managing state machines.
2.  A UI for visualizing execution and debugging.
3.  Integrations for persistence, telemetry, and system integration.

## What Can You Build with Burr?

Burr supports diverse applications:

*   [GPT-like chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/multi-modal-chatbot)
*   [Stateful RAG-based chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/conversational-rag/simple_example)
*   [LLM-based adventure game](https://github.com/DAGWorks-Inc/burr/tree/main/examples/llm-adventure-game)
*   [Interactive email assistant](https://github.com/DAGWorks-Inc/burr/tree/main/examples/email-assistant)
*   [Simulation](https://github.com/DAGWorks-Inc/burr/tree/main/examples/simulation)
*   [Hyperparameter tuning](https://github.com/DAGWorks-Inc/burr/tree/main/examples/ml-training)

## Roadmap

Burr is continually evolving, with future features including:

*   FastAPI integration and hosted deployment.
*   Improvements to the core library (e.g., retries, exception management).
*   Enhanced integration with popular frameworks (LCEL, LlamaIndex, etc.).
*   Hosted execution tooling.
*   Additional storage integrations.
*   Burr Cloud (hosted solution, sign up [here](https://forms.gle/w9u2QKcPrztApRedA) for the waitlist).

## Contribute

We welcome contributions! Refer to the [developer documentation](https://burr.dagworks.io/contributing) to get started.

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

## Contributors

### Code contributions

Users who have contributed core functionality, integrations, or examples.

- [Elijah ben Izzy](https://github.com/elijahbenizzy)
- [Stefan Krawczyk](https://github.com/skrawcz)
- [Joseph Booth](https://github.com/jombooth)
- [Nandani Thakur](https://github.com/NandaniThakur)
- [Thierry Jean](https://github.com/zilto)
- [Hamza Farhan](https://github.com/HamzaFarhan)
- [Abdul Rafay](https://github.com/proftorch)
- [Margaret Lange](https://github.com/margaretlange)

### Bug hunters/special mentions

Users who have contributed small docs fixes, design suggestions, and found bugs

- [Luke Chadwick](https://github.com/vertis)
- [Evans](https://github.com/sudoevans)
- [Sasmitha Manathunga](https://github.com/mmz-001)