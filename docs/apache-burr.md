# Burr: Build Stateful AI Applications with Ease

Burr empowers developers to create robust, stateful AI applications, chatbots, and more, using simple Python building blocks.  [View the original repository on GitHub](https://github.com/apache/burr).

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

*   **State Machine Approach:**  Model your application as a state machine for clear, manageable logic.
*   **Python-First Development:** Build with simple Python functions, minimizing dependencies.
*   **Real-time UI:**  Monitor, trace, and debug your application's execution with a built-in user interface.
*   **Pluggable Persistence:** Easily save and load application state with flexible persister integrations.
*   **Framework Agnostic:** Seamlessly integrates with your favorite frameworks and tools.
*   **Versatile Applications:** Suitable for chatbots, agents, simulations, and more.
*   **Open-Source & Community Driven:** Benefit from a community and contribute to the project.

## Getting Started

### Installation

Install Burr using pip:

```bash
pip install "burr[start]"
```

*(See the [documentation](https://burr.dagworks.io/getting_started/install/) if using poetry.)*

### Run the UI

Start the Burr UI server:

```bash
burr
```

The UI will open with sample data and a demo chatbot.

### Example:

1.  Clone the repository and navigate to the hello-world counter example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

2.  Observe the counter example and its trace in the UI.

### More Information

*   [Documentation](https://burr.dagworks.io/)
*   [Quick Intro Video](https://www.loom.com/share/a10f163428b942fea55db1a84b1140d8?sid=1512863b-f533-4a42-a2f3-95b13deb07c9)
*   [Longer Video Intro & Walkthrough](https://www.youtube.com/watch?v=rEZ4oDN0GdU)
*   [Blog Post](https://blog.dagworks.io/p/burr-develop-stateful-ai-applications)
*   [Discord for Help/Questions](https://discord.gg/6Zy2DwP4f3)

## How Burr Works

Burr uses a simple API to define your application as a state machine.  It offers clear structure for complex workflows.

**Core Example:**

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

Burr provides:

1.  A lightweight Python library for building and managing state machines.
2.  A UI for execution telemetry, introspection, and debugging.
3.  Integrations for state persistence, telemetry connections, and other systems.

![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## Use Cases

Burr can power diverse applications:

1.  [Simple GPT-like chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/multi-modal-chatbot)
2.  [Stateful RAG-based chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/conversational-rag/simple_example)
3.  [LLM-based adventure game](https://github.com/DAGWorks-Inc/burr/tree/main/examples/llm-adventure-game)
4.  [Interactive assistant for writing emails](https://github.com/DAGWorks-Inc/burr/tree/main/examples/email-assistant)

Additionally, Burr supports non-LLM use cases like a [simulation](https://github.com/DAGWorks-Inc/burr/tree/main/examples/simulation) and [hyperparameter tuning](https://github.com/dagworks-inc/burr/tree/main/examples/ml-training).

Burr integrates with various vendors and libraries to build custom actions.

## Comparison with Other Frameworks

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why the Name "Burr"?

Named after Aaron Burr, the project's namesake reflects the connection to Hamilton.  Burr started as a tool to handle state between Hamilton DAG executions.

## Testimonials

*   *[Quotes from Ashish Ghosh, CTO of Peanut Robotics]*
*   *[Quotes from Reddit user cyan2k of LocalLlama Subreddit]*
*   *[Quotes from Ishita, Founder of Watto.ai]*
*   *[Quotes from Matthew Rideout, Staff Software Engineer, Paxton AI]*
*   *[Quotes from Rinat Gareev, Senior Solutions Architect, Provectus]*
*   *[Quotes from Hadi Nayebi, Co-founder, CognitiveGraphs]*
*   *[Quotes from Aditya K., DS Architect, TaskHuman]*

## Roadmap

Planned features include:

1.  FastAPI integration and hosted deployment.
2.  Core library improvements: retries, exception management, integration with frameworks (LCEL, LlamaIndex, Hamilton), metadata capture, pydantic-based typing improvements.
3.  Tooling for hosted execution integrating with your infrastructure.
4.  Additional storage integrations.

Sign up [here](https://forms.gle/w9u2QKcPrztApRedA) for the Burr Cloud waitlist.

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

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

### Bug Hunters/Special Mentions

*   [Luke Chadwick](https://github.com/vertis)
*   [Evans](https://github.com/sudoevans)
*   [Sasmitha Manathunga](https://github.com/mmz-001)