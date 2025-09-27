<div align="center">
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01nTg6w21NqT5qFKH1u_!!6000000001621-55-tps-550-550.svg" alt="AgentScope Logo" width="200">
  <h1>AgentScope: Build LLM Applications with Ease</h1>
</div>

<p align="center">
  AgentScope empowers developers to build powerful and customizable LLM applications through a flexible and agent-oriented programming model.  <a href="https://github.com/agentscope-ai/agentscope">Explore the code on GitHub</a>!
</p>

<div align="center">
  <a href="https://arxiv.org/abs/2402.14034">
    <img src="https://img.shields.io/badge/cs.MA-2402.14034-B31C1C?logo=arxiv&logoColor=B31C1C" alt="arxiv">
  </a>
  <a href="https://pypi.org/project/agentscope/">
    <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python" alt="pypi">
  </a>
  <a href="https://pypi.org/project/agentscope/">
    <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fagentscope%2Fjson&query=%24.info.version&prefix=v&logo=pypi&label=version" alt="pypi">
  </a>
  <a href="https://doc.agentscope.io/">
    <img src="https://img.shields.io/badge/Docs-English%7C%E4%B8%AD%E6%96%87-blue?logo=markdown" alt="docs">
  </a>
  <a href="https://agentscope.io/">
    <img src="https://img.shields.io/badge/GUI-AgentScope_Studio-blue?logo=look&logoColor=green&color=dark-green" alt="workstation">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-black" alt="license">
  </a>
  <img src="https://trendshift.io/api/badge/repositories/10079" alt="modelscope%2Fagentscope | Trendshift" style="width: 250px; height: 55px;" width="250" height="55">
</div>

## Key Features

*   **Transparent and Controllable:**  AgentScope prioritizes transparency, giving developers full control over prompt engineering, API calls, agent construction, and workflow orchestration.
*   **Realtime Steering:** Offers native support for real-time interruption and customized handling of agent actions.
*   **Agentic Capabilities:** Enhanced support for agentic tools management, long-term memory control, and Retrieval-Augmented Generation (RAG) strategies.
*   **Model Agnostic:**  Write your code once and run it with various LLM models.
*   **Modular Design:** Build agents using LEGO-style, independent, and modular components.
*   **Multi-Agent Focused:** Designed for multi-agent collaboration with explicit message passing and workflow orchestration, avoiding deep encapsulation.
*   **Highly Customizable:**  Offers extensive customization options for tools, prompts, agents, workflows, third-party libraries, and visualizations.

## What's New in AgentScope v1.0?

The latest version of AgentScope introduces significant improvements:

*   **Voice Agent:** New voice agent support using ReActAgent and native support for Qwen-Omni and GPT-Audio. See the [new example](https://github.com/agentscope-ai/agentscope/tree/main/examples/agent/voice_agent).
*   **Plan Module:**  A new powerful 📋**Plan** module is now available. Learn more in the [tutorial](https://doc.agentscope.io/tutorial/task_plan.html).
*   **AgentScope Runtime:** Open-sourced for effective agent deployment, sandboxed tool execution, and production-ready AI applications. Explore the [GitHub repo](https://github.com/agentscope-ai/agentscope-runtime).
*   **AgentScope Studio:** Open-sourced for tracing and visualizing your agent applications. Explore the [GitHub repo](https://github.com/agentscope-ai/agentscope-studio).
*   **Enhanced Features:**  Fully embraces asynchronous execution and introduces numerous new features and improvements.  See the [changelog](https://github.com/agentscope-ai/agentscope/blob/main/docs/changelog.md) for detailed changes.

## Table of Contents

-   [🚀 Quickstart](#-quickstart)
    -   [💻 Installation](#-installation)
        -   [🛠️ From Source](#-from-source)
        -   [📦 From PyPi](#-from-pypi)
-   [📝 Example](#-example)
    -   [👋 Hello AgentScope!](#-hello-agentscope)
    -   [🎯 Realtime Steering](#-realtime-steering)
    -   [🛠️ Fine-Grained MCP Control](#-fine-grained-mcp-control)
    -   [🧑‍🤝‍🧑 Multi-Agent Conversation](#-multi-agent-conversation)
    -   [💻 AgentScope Studio](#-agentscope-studio)
-   [📖 Documentation](#-documentation)
-   [⚖️ License](#-license)
-   [📚 Publications](#-publications)
-   [✨ Contributors](#-contributors)

## 🚀 Quickstart

### 💻 Installation

> AgentScope requires **Python 3.10** or higher.

#### 🛠️ From Source

```bash
# Clone the repository
git clone -b main https://github.com/agentscope-ai/agentscope.git
cd agentscope

# Install in editable mode
pip install -e .
```

#### 📦 From PyPi

```bash
pip install agentscope
```

## 📝 Example

### 👋 Hello AgentScope!

Get started with a conversation between a user and a ReAct agent 🤖 named "Friday"!

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

### 🎯 Realtime Steering

Benefit from **realtime interruption** in ``ReActAgent`` for enhanced control with robust memory preservation. Interruptions become **observable events** for seamless conversation resumption.

<div align="center">
  <img src="./assets/images/realtime_steering_zh.gif" alt="Realtime Steering" width="49%">
  <img src="./assets/images/realtime_steering_en.gif" alt="Realtime Steering" width="49%">
</div>

### 🛠️ Fine-Grained MCP Control

Developers can get the MCP tool as a **local callable function**, using it directly, passing to an agent, or incorporating into a more complex tool.

```python
from agentscope.mcp import HttpStatelessClient
from agentscope.tool import Toolkit
import os

async def fine_grained_mcp_control():
    # Initialize the MCP client
    client = HttpStatelessClient(
        name="gaode_mcp",
        transport="streamable_http",
        url=f"https://mcp.amap.com/mcp?key={os.environ['GAODE_API_KEY']}",
    )

    # Get the MCP tool as a local callable function.
    func = await client.get_callable_function(func_name="maps_geo")

    # Option 1: Call directly
    await func(address="Tiananmen Square", city="Beijing")

    # Option 2: Pass to agent as a tool
    toolkit = Toolkit()
    toolkit.register_tool_function(func)
    # ...

    # Option 3: Wrap into a more complex tool
    # ...
```

### 🧑‍🤝‍🧑 Multi-Agent Conversation

AgentScope provides ``MsgHub`` and pipelines to streamline multi-agent conversations, allowing efficient message routing and seamless information sharing.

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

### 💻 AgentScope Studio

Use the following commands to install and start AgentScope Studio for tracing and visualizing your agent application:

```bash
npm install -g @agentscope/studio
as_studio
```

<div align="center">
    <img src="./assets/images/home.gif" width="49%" alt="home">
    <img src="./assets/images/projects.gif" width="49%" alt="projects">
    <img src="./assets/images/runtime.gif" width="49%" alt="runtime">
    <img src="./assets/images/friday.gif" width="49%" alt="friday">
</div>

## 📖 Documentation

*   **Tutorial**
    *   [Installation](https://doc.agentscope.io/tutorial/quickstart_installation.html)
    *   [Key Concepts](https://doc.agentscope.io/tutorial/quickstart_key_concept.html)
    *   [Create Message](https://doc.agentscope.io/tutorial/quickstart_message.html)
    *   [ReAct Agent](https://doc.agentscope.io/tutorial/quickstart_agent.html)
*   **Workflow**
    *   [Conversation](https://doc.agentscope.io/tutorial/workflow_conversation.html)
    *   [Multi-Agent Debate](https://doc.agentscope.io/tutorial/workflow_multiagent_debate.html)
    *   [Concurrent Agents](https://doc.agentscope.io/tutorial/workflow_concurrent_agents.html)
    *   [Routing](https://doc.agentscope.io/tutorial/workflow_routing.html)
    *   [Handoffs](https://doc.agentscope.io/tutorial/workflow_handoffs.html)
*   **FAQ**
    *   [FAQ](https://doc.agentscope.io/tutorial/faq.html)
*   **Task Guides**
    *   [Model](https://doc.agentscope.io/tutorial/task_model.html)
    *   [Prompt Formatter](https://doc.agentscope.io/tutorial/task_prompt.html)
    *   [Tool](https://doc.agentscope.io/tutorial/task_tool.html)
    *   [Memory](https://doc.agentscope.io/tutorial/task_memory.html)
    *   [Long-Term Memory](https://doc.agentscope.io/tutorial/task_long_term_memory.html)
    *   [Agent](https://doc.agentscope.io/tutorial/task_agent.html)
    *   [Pipeline](https://doc.agentscope.io/tutorial/task_pipeline.html)
    *   [Plan](https://doc.agentscope.io/tutorial/task_plan.html)
    *   [State/Session Management](https://doc.agentscope.io/tutorial/task_state.html)
    *   [Agent Hooks](https://doc.agentscope.io/tutorial/task_hook.html)
    *   [MCP](https://doc.agentscope.io/tutorial/task_mcp.html)
    *   [AgentScope Studio](https://doc.agentscope.io/tutorial/task_studio.html)
    *   [Tracing](https://doc.agentscope.io/tutorial/task_tracing.html)
    *   [Evaluation](https://doc.agentscope.io/tutorial/task_eval.html)
    *   [Embedding](https://doc.agentscope.io/tutorial/task_embedding.html)
    *   [Token](https://doc.agentscope.io/tutorial/task_token.html)
*   **API**
    *   [API Docs](https://doc.agentscope.io/api/agentscope.html)
*   **Examples**
    *   Game
        *   [Nine-player Werewolves](https://github.com/agentscope-ai/agentscope/tree/main/examples/game/werewolves)
    *   Workflow
        *   [Multi-agent Debate](https://github.com/agentscope-ai/agentscope/tree/main/examples/workflows/multiagent_debate)
        *   [Multi-agent Conversation](https://github.com/agentscope-ai/agentscope/tree/main/examples/workflows/multiagent_conversation)
        *   [Multi-agent Concurrent](https://github.com/agentscope-ai/agentscope/tree/main/examples/workflows/multiagent_concurrent)
    *   Evaluation
        *   [ACEBench](https://github.com/agentscope-ai/agentscope/tree/main/examples/evaluation/ace_bench)
    *   Functional
        *   [ReAct Agent](https://github.com/agentscope-ai/agentscope/tree/main/examples/react_agent)
        *   [MCP](https://github.com/agentscope-ai/agentscope/tree/main/examples/functionality/mcp)
        *   [Plan](https://github.com/agentscope-ai/agentscope/tree/main/examples/functionality/plan)
        *   [Structured Output](https://github.com/agentscope-ai/agentscope/tree/main/examples/functionality/structured_output)
        *   [Long-Term Memory](https://github.com/agentscope-ai/agentscope/tree/main/examples/functionality/long_term_memory)
        *   [Session with SQLite](https://github.com/agentscope-ai/agentscope/tree/main/examples/functionality/session_with_sqlite)
        *   [Voice Agent](https://github.com/agentscope-ai/agentscope/tree/main/examples/agent/voice_agent)

## ⚖️ License

AgentScope is released under the Apache License 2.0.

## 📚 Publications

If you find our work helpful for your research or application, please cite our papers:

-   [AgentScope 1.0: A Developer-Centric Framework for Building Agentic Applications](https://arxiv.org/abs/2508.16279)

-   [AgentScope: A Flexible yet Robust Multi-Agent Platform](https://arxiv.org/abs/2402.14034)

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

## ✨ Contributors

A big thank you to all our contributors:

<a href="https://github.com/agentscope-ai/agentscope/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=agentscope-ai/agentscope&max=999&columns=12&anon=1" />
</a>