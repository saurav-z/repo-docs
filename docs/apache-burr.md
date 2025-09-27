# Burr: Build and Monitor State Machines for AI Applications

Burr simplifies the development of decision-making applications, empowering you to build robust chatbots, agents, and simulations using simple Python building blocks. [Visit the original repo](https://github.com/apache/burr)

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

*   **Simplified State Management:** Easily define and manage complex workflows using state machines.
*   **Framework Agnostic:** Burr integrates seamlessly with your favorite LLMs and frameworks.
*   **Real-time Monitoring:** Built-in UI for tracking, monitoring, and tracing your system's execution.
*   **Pluggable Persisters:** Save and load application state with various persistence options.
*   **Open-Source & Extensible:** Leverage a growing ecosystem and contribute to the community.

## Getting Started

### Installation

Install Burr using pip:

```bash
pip install "burr[start]"
```

### Run the UI
Then run the UI server:

```bash
burr
```
This will open up Burr's telemetry UI.  It comes loaded with some default data so you can click around.
It also has a demo chat application to help demonstrate what the UI captures enabling you too see things changing in
real-time. Hit the "Demos" side bar on the left and select `chatbot`. To chat it requires the `OPENAI_API_KEY`
environment variable to be set, but you can still see how it works if you don't have an API key set.

### Example
Next, start coding / running examples:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

You'll see the counter example running in the terminal, along with the trace being tracked in the UI.
See if you can find it.

For more details see the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr allows you to express your application logic as a state machine, defined by simple Python functions and transitions. It simplifies complex decision-making processes.

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

1.  A lightweight Python library for building and managing state machines.
2.  A UI for execution telemetry, enabling introspection and debugging.
3.  Integrations for state persistence, telemetry, and connection with other systems.

![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## Use Cases

Burr is ideal for a variety of applications:

*   Chatbots (simple and RAG-based)
*   LLM-based adventure games
*   Interactive assistants (e.g., email writing)
*   Simulations (e.g., time-series forecasting)
*   Hyperparameter tuning and more!

## Comparisons with Other Frameworks

Burr offers a unique approach, offering a dedicated focus on state machines.

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Contributing
We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

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