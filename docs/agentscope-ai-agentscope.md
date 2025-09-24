# AgentScope: Build LLM Applications with Agent-Oriented Programming

AgentScope is a powerful, flexible, and developer-friendly framework designed to simplify the creation of Large Language Model (LLM) applications using an agent-oriented programming paradigm.  [Explore the original repository](https://github.com/agentscope-ai/agentscope).

<p align="center">
  <img
    src="https://img.alicdn.com/imgextra/i1/O1CN01nTg6w21NqT5qFKH1u_!!6000000001621-55-tps-550-550.svg"
    alt="AgentScope Logo"
    width="200"
  />
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2402.14034">
        <img
            src="https://img.shields.io/badge/cs.MA-2402.14034-B31C1C?logo=arxiv&logoColor=B31C1C"
            alt="arxiv"
        />
    </a>
    <a href="https://pypi.org/project/agentscope/">
        <img
            src="https://img.shields.io/badge/python-3.10+-blue?logo=python"
            alt="pypi"
        />
    </a>
    <a href="https://pypi.org/project/agentscope/">
        <img
            src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fagentscope%2Fjson&query=%24.info.version&prefix=v&logo=pypi&label=version"
            alt="pypi"
        />
    </a>
    <a href="https://doc.agentscope.io/">
        <img
            src="https://img.shields.io/badge/Docs-English%7C%E4%B8%AD%E6%96%87-blue?logo=markdown"
            alt="docs"
        />
    </a>
    <a href="https://agentscope.io/">
        <img
            src="https://img.shields.io/badge/GUI-AgentScope_Studio-blue?logo=look&logoColor=green&color=dark-green"
            alt="workstation"
        />
    </a>
    <a href="./LICENSE">
        <img
            src="https://img.shields.io/badge/license-Apache--2.0-black"
            alt="license"
        />
    </a>
</p>

<p align="center">
<img src="https://trendshift.io/api/badge/repositories/10079" alt="modelscope%2Fagentscope | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</p>

## Key Features of AgentScope

AgentScope empowers developers with features for building robust and flexible LLM applications:

*   **Transparent & Controllable:**  Provides full visibility and control over prompt engineering, API calls, agent construction, and workflow orchestration.
*   **Realtime Steering:** Native support for real-time interruption and customized handling, improving user interaction.
*   **Agentic Capabilities:** Offers advanced features like agentic tools management, long-term memory control, and RAG (Retrieval-Augmented Generation).
*   **Model Agnostic:**  Write your code once and run it with various LLM models.
*   **Modular Design:**  Employs a LEGO-style approach, where components are modular and independent for easy customization.
*   **Multi-Agent Focused:** Designed for multi-agent systems with explicit message passing and clear workflow orchestration.
*   **Highly Customizable:**  Allows customization of tools, prompts, agents, workflows, third-party libraries, and visualizations.

## AgentScope 1.0 Key Features

| Module     | Feature                                                                            | Tutorial                                                                |
|------------|------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| model      | Support async invocation                                                           | [Model](https://doc.agentscope.io/tutorial/task_model.html)             |
|            | Support reasoning model                                                            |                                                                         |
|            | Support streaming/non-streaming returns                                            |                                                                         |
| tool       | Support async/sync tool functions                                                  | [Tool](https://doc.agentscope.io/tutorial/task_tool.html)               |
|            | Support streaming/non-streaming returns                                            |                                                                         |
|            | Support user interruption                                                          |                                                                         |
|            | Support post-processing                                                            |                                                                         |
|            | Support group-wise tools management                                                |                                                                         |
|            | Support agentic tools management by meta tool                                      |                                                                         |
| MCP        | Support streamable HTTP/SSE/StdIO transport                                        | [MCP](https://doc.agentscope.io/tutorial/task_mcp.html)                 |
|            | Support both **stateful** and **stateless** mode MCP Client                        |                                                                         |
|            | Support client- & function-level fine-grained control                              |                                                                         |
| agent      | Support async execution                                                            |                                                                         |
|            | Support parallel tool calls                                                        |                                                                         |
|            | Support realtime steering interruption and customized handling                     |                                                                         |
|            | Support automatic state management                                                 |                                                                         |
|            | Support agent-controlled long-term memory                                          |                                                                         |
|            | Support agent hooks                                                                |                                                                         |
| tracing    | Support OpenTelemetry-based tracing in LLM, tools, agent and formatter             | [Tracing](https://doc.agentscope.io/tutorial/task_tracing.html)         |
|            | Support connecting to third-party tracing platforms (e.g. Arize-Phoenix, Langfuse) |                                                                         |
| memory     | Support long-term memory                                                           | [Memory](https://doc.agentscope.io/tutorial/task_long_term_memory.html) |
| session    | Provide session/application-level automatic state management                       | [Session](https://doc.agentscope.io/tutorial/task_state.html)           |
| evaluation | Provide distributed and parallel evaluation                                        | [Evaluation](https://doc.agentscope.io/tutorial/task_eval.html)         |
| formatter  | Support multi-agent prompt formatting with tools API                               | [Prompt Formatter](https://doc.agentscope.io/tutorial/task_prompt.html) |
|            | Support truncation-based formatter strategy                                        |                                                                         |
| plan       | Support ReAct-based long-term planning                                             | [Plan](https://doc.agentscope.io/tutorial/task_plan.html)               |
|            | Support manual plan specification                                                 |                                                                         |
| ...        |                                                                                    |                                                                         |

## News

*   **\[2025-09]** **Voice agent** is online! `ReActAgent` supports Qwen-Omni and GPT-Audio natively now, check our [new example](https://github.com/agentscope-ai/agentscope/tree/main/examples/agent/voice_agent) and [roadmap](https://github.com/agentscope-ai/agentscope/issues/773).
*   **\[2025-09]** A new powerful 📋**Plan** module is online now! Check out the [tutorial](https://doc.agentscope.io/tutorial/task_plan.html) for more details.
*   **\[2025-09]** **AgentScope Runtime** is open-sourced now! Enabling effective agent deployment with sandboxed tool execution for production-ready AI applications. Check out the [GitHub repo](https://github.com/agentscope-ai/agentscope-runtime).
*   **\[2025-09]** **AgentScope Studio** is open-sourced now! Check out the [GitHub repo](https://github.com/agentscope-ai/agentscope-studio).
*   **\[2025-08]** The new tutorial of v1 is online now! Check out the [tutorial](https://doc.agentscope.io) for more details.
*   **\[2025-08]** 🎉🎉 AgentScope v1 is released now! This version fully embraces the asynchronous execution, providing many new features and improvements. Check out [changelog](https://github.com/agentscope-ai/agentscope/blob/main/docs/changelog.md) for detailed changes.

## Quickstart Guide

### Installation

**Prerequisites:**  AgentScope requires Python 3.10 or higher.

#### From Source

```bash
git clone -b main https://github.com/agentscope-ai/agentscope.git
cd agentscope
pip install -e .
```

#### From PyPi

```bash
pip install agentscope
```

## Examples

### Hello AgentScope!

```python
from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, execute_python_code, execute_shell_command
import os, asyncio


async def main():
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    toolkit.register_tool_function(execute_shell_command)

    agent = ReActAgent(
        name="Friday",
        sys_prompt="You're a helpful assistant named Friday.",
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=True,
        ),
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
    )

    user = UserAgent(name="user")

    msg = None
    while True:
        msg = await agent(msg)
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break

asyncio.run(main())
```

### Realtime Steering

<p align="center">
  <img src="./assets/images/realtime_steering_zh.gif" alt="Realtime Steering" width="49%"/>
  <img src="./assets/images/realtime_steering_en.gif" alt="Realtime Steering" width="49%"/>
</p>

### Fine-Grained MCP Control

```python
from agentscope.mcp import HttpStatelessClient
from agentscope.tool import Toolkit
import os

async def fine_grained_mcp_control():
    client = HttpStatelessClient(
        name="gaode_mcp",
        transport="streamable_http",
        url=f"https://mcp.amap.com/mcp?key={os.environ['GAODE_API_KEY']}",
    )
    func = await client.get_callable_function(func_name="maps_geo")
    await func(address="Tiananmen Square", city="Beijing")
    toolkit = Toolkit()
    toolkit.register_tool_function(func)
```

### Multi-Agent Conversation

```python
from agentscope.pipeline import MsgHub, sequential_pipeline
from agentscope.message import Msg
import asyncio

async def multi_agent_conversation():
    # Create agents
    agent1 = ...
    agent2 = ...
    agent3 = ...
    agent4 = ...

    # Create a message hub to manage multi-agent conversation
    async with MsgHub(
        participants=[agent1, agent2, agent3],
        announcement=Msg("Host", "Introduce yourselves.", "assistant")
    ) as hub:
        # Speak in a sequential manner
        await sequential_pipeline([agent1, agent2, agent3])
        # Dynamic manage the participants
        hub.add(agent4)
        hub.delete(agent3)
        await hub.broadcast(Msg("Host", "Goodbye!", "assistant"))

asyncio.run(multi_agent_conversation())
```

### AgentScope Studio

```bash
npm install -g @agentscope/studio
as_studio
```

<p align="center">
    <img
        src="./assets/images/home.gif"
        width="49%"
        alt="home"
    />
    <img
        src="./assets/images/projects.gif"
        width="49%"
        alt="projects"
    />
    <img
        src="./assets/images/runtime.gif"
        width="49%"
        alt="runtime"
    />
    <img
        src="./assets/images/friday.gif"
        width="49%"
        alt="friday"
    />
</p>

## Documentation

*   [Tutorial](https://doc.agentscope.io/)
*   [API Docs](https://doc.agentscope.io/api/agentscope.html)
*   [Examples](https://github.com/agentscope-ai/agentscope/tree/main/examples)

## License

AgentScope is released under the Apache License 2.0.

## Publications

*   [AgentScope 1.0: A Developer-Centric Framework for Building Agentic Applications](https://arxiv.org/abs/2508.16279)

*   [AgentScope: A Flexible yet Robust Multi-Agent Platform](https://arxiv.org/abs/2402.14034)

```
@article{agentscope_v1,
    author  = {
        Dawei Gao,
        Zitao Li,
        Yuexiang Xie,
        Weirui Kuang,
        Liuyi Yao,
        Bingchen Qian,
        Zhijian Ma,
        Yue Cui,
        Haohao Luo,
        Shen Li,
        Lu Yi,
        Yi Yu,
        Shiqi He,
        Zhiling Luo,
        Wenmeng Zhou,
        Zhicheng Zhang,
        Xuguang He,
        Ziqian Chen,
        Weikai Liao,
        Farruh Isakulovich Kushnazarov,
        Yaliang Li,
        Bolin Ding,
        Jingren Zhou}
    title   = {AgentScope 1.0: A Developer-Centric Framework for Building Agentic Applications},
    journal = {CoRR},
    volume  = {abs/2508.16279},
    year    = {2025},
}

@article{agentscope,
    author  = {
        Dawei Gao,
        Zitao Li,
        Xuchen Pan,
        Weirui Kuang,
        Zhijian Ma,
        Bingchen Qian,
        Fei Wei,
        Wenhao Zhang,
        Yuexiang Xie,
        Daoyuan Chen,
        Liuyi Yao,
        Hongyi Peng,
        Zeyu Zhang,
        Lin Zhu,
        Chen Cheng,
        Hongzhu Shi,
        Yaliang Li,
        Bolin Ding,
        Jingren Zhou}
    title   = {AgentScope: A Flexible yet Robust Multi-Agent Platform},
    journal = {CoRR},
    volume  = {abs/2402.14034},
    year    = {2024},
}
```

## Contributors

<a href="https://github.com/agentscope-ai/agentscope/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=agentscope-ai/agentscope&max=999&columns=12&anon=1" />
</a>