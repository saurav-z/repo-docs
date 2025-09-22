<p align="center">
  <img
    src="https://img.alicdn.com/imgextra/i1/O1CN01nTg6w21NqT5qFKH1u_!!6000000001621-55-tps-550-550.svg"
    alt="AgentScope Logo"
    width="200"
  />
</p>

<h2 align="center">AgentScope: Build Powerful LLM Applications with Agent-Oriented Programming</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2402.14034">
    <img
      src="https://img.shields.io/badge/cs.MA-2402.14034-B31C1C?logo=arxiv&logoColor=B31C1C"
      alt="arXiv"
    />
  </a>
  <a href="https://pypi.org/project/agentscope/">
    <img
      src="https://img.shields.io/badge/python-3.10+-blue?logo=python"
      alt="PyPI"
    />
  </a>
  <a href="https://pypi.org/project/agentscope/">
    <img
      src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fagentscope%2Fjson&query=%24.info.version&prefix=v&logo=pypi&label=version"
      alt="Version"
    />
  </a>
  <a href="https://doc.agentscope.io/">
    <img
      src="https://img.shields.io/badge/Docs-English%7C%E4%B8%AD%E6%96%87-blue?logo=markdown"
      alt="Documentation"
    />
  </a>
  <a href="https://agentscope.io/">
    <img
      src="https://img.shields.io/badge/GUI-AgentScope_Studio-blue?logo=look&logoColor=green&color=dark-green"
      alt="AgentScope Studio"
    />
  </a>
  <a href="./LICENSE">
    <img
      src="https://img.shields.io/badge/license-Apache--2.0-black"
      alt="License"
    />
  </a>
</p>

<p align="center">
<img src="https://trendshift.io/api/badge/repositories/10079" alt="modelscope%2Fagentscope | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</p>

AgentScope empowers developers to build and deploy sophisticated LLM-powered applications using an intuitive, agent-oriented programming paradigm. [Check out the original repo](https://github.com/agentscope-ai/agentscope) for the latest updates and to contribute!

## Key Features of AgentScope

*   **Transparent Development:**  Gain full control over prompt engineering, API calls, agent construction, and workflow orchestration.
*   **Realtime Steering:**  Enable dynamic interruption and customization within your agent workflows.
*   **Agentic Capabilities:**  Manage tools, long-term memory, and Retrieval Augmented Generation (RAG) effectively.
*   **Model Agnostic:**  Write code once and run it with various LLM models.
*   **Modular Design:**  Build agents using independent, LEGO-style components.
*   **Multi-Agent Focus:**  Design explicit message passing and orchestration for multi-agent systems.
*   **Highly Customizable:**  Tailor tools, prompts, agents, workflows, and visualizations to your specific needs.

## What's New in AgentScope 1.0

*   **Voice Agent**: New support for voice agents with Qwen-Omni and GPT-Audio natively.
*   **Plan Module**: New Plan module for ReAct-based long-term planning.
*   **AgentScope Runtime**: Open-sourced for effective agent deployment.
*   **AgentScope Studio**: Open-sourced for tracing and visualization.
*   **Asynchronous Execution**: Fully embraces async execution for better performance and responsiveness.

## Table of Contents

-   [🚀 Quickstart](#-quickstart)
    -   [💻 Installation](#-installation)
        -   [🛠️ From Source](#-from-source)
        -   [📦 From PyPi](#-from-pypi)
-   [📝 Examples](#-example)
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
# Pull the source code from GitHub
git clone -b main https://github.com/agentscope-ai/agentscope.git

# Install the package in editable mode
cd agentscope
pip install -e .
```

#### 📦 From PyPi

```bash
pip install agentscope
```

## 📝 Examples

### 👋 Hello AgentScope!

A simple conversation between a user and a ReAct agent named "Friday".

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

Demonstrates native support for **realtime interruption** in `ReActAgent` with robust memory preservation, converting interruption into an **observable event**.

<p align="center">
  <img src="./assets/images/realtime_steering_zh.gif" alt="Realtime Steering" width="49%"/>
  <img src="./assets/images/realtime_steering_en.gif" alt="Realtime Steering" width="49%"/>
</p>

### 🛠️ Fine-Grained MCP Control

Shows how developers can obtain MCP tools as **local callable functions**, enabling flexible integration.

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

    # Obtain the MCP tool as a **local callable function**, and use it anywhere
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

Leverage `MsgHub` and pipelines to create efficient multi-agent conversations.

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

Start AgentScope Studio with:

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

## 📖 Documentation

*   [Tutorial](https://doc.agentscope.io/) - Get started quickly!
*   [API Reference](https://doc.agentscope.io/api/agentscope.html)
*   [Examples](https://github.com/agentscope-ai/agentscope/tree/main/examples)

## ⚖️ License

AgentScope is released under the Apache License 2.0.

## 📚 Publications

If you find our work helpful, please cite our papers:

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

## ✨ Contributors

Thank you to all our contributors!

<a href="https://github.com/agentscope-ai/agentscope/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=agentscope-ai/agentscope&max=999&columns=12&anon=1" />
</a>