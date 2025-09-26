<!-- Improved & SEO-Optimized README for AgentScope -->
<!-- Link back to original repo -->
[**Original Repository: AgentScope**](https://github.com/agentscope-ai/agentscope)

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

AgentScope is a powerful, developer-friendly framework for building, deploying, and managing sophisticated Large Language Model (LLM) applications using an agent-oriented programming paradigm.

## Key Features of AgentScope

*   **Transparent Development:** Full control and visibility over prompt engineering, API calls, agent creation, and workflow orchestration.
*   **Realtime Steering:**  Native support for real-time interruption and customizable handling.
*   **Agentic Capabilities:** Integrated support for agentic tools management, long-term memory, and Retrieval-Augmented Generation (RAG).
*   **Model Agnostic:** Develop once, run with various LLM models.
*   **Modular Design:**  Build agents using independent, LEGO-style components.
*   **Multi-Agent Focus:** Designed for multi-agent systems, explicit message passing, and seamless workflow orchestration.
*   **Highly Customizable:** Easy to customize tools, prompts, agents, workflows, and third-party integrations.

<p align="center">
<img src="./assets/images/agentscope_v1_0822.png" alt="AgentScope Framework" width="80%"/>
</p>

## Key Modules & Features in AgentScope 1.0:

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

*   **[2025-09] Voice Agent:**  Now supports Qwen-Omni and GPT-Audio natively.  Check the [example](https://github.com/agentscope-ai/agentscope/tree/main/examples/agent/voice_agent) and [roadmap](https://github.com/agentscope-ai/agentscope/issues/773).
*   **[2025-09] New Plan Module:** Powerful new Plan module is now available.  See the [tutorial](https://doc.agentscope.io/tutorial/task_plan.html).
*   **[2025-09] AgentScope Runtime:** Open-sourced for effective agent deployment with sandboxed tool execution.  [GitHub repo](https://github.com/agentscope-ai/agentscope-runtime).
*   **[2025-09] AgentScope Studio:** Open-sourced for tracing and visualization.  [GitHub repo](https://github.com/agentscope-ai/agentscope-studio).
*   **[2025-08] v1 Tutorial:** Updated tutorial is available at [doc.agentscope.io](https://doc.agentscope.io).
*   **[2025-08] AgentScope v1 Released:**  Fully embraces asynchronous execution. See [changelog](https://github.com/agentscope-ai/agentscope/blob/main/docs/changelog.md).

## Contact

Join the AgentScope community:

| [Discord](https://discord.gg/eYMpfnkG8h)                                                                                         | DingTalk                                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://gw.alicdn.com/imgextra/i1/O1CN01hhD1mu1Dd3BWVUvxN_!!6000000000238-2-tps-400-400.png" width="100" height="100"> | <img src="https://img.alicdn.com/imgextra/i1/O1CN01LxzZha1thpIN2cc2E_!!6000000005934-2-tps-497-477.png" width="100" height="100"> |

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

## 📑 Table of Contents

- [🚀 Quickstart](#-quickstart)
    - [💻 Installation](#-installation)
        - [🛠️ From Source](#-from-source)
        - [📦 From PyPi](#-from-pypi)
- [📝 Example](#-example)
    - [👋 Hello AgentScope!](#-hello-agentscope)
    - [🎯 Realtime Steering](#-realtime-steering)
    - [🛠️ Fine-Grained MCP Control](#-fine-grained-mcp-control)
    - [🧑‍🤝‍🧑 Multi-Agent Conversation](#-multi-agent-conversation)
    - [💻 AgentScope Studio](#-agentscope-studio)
- [📖 Documentation](#-documentation)
- [⚖️ License](#-license)
- [📚 Publications](#-publications)
- [✨ Contributors](#-contributors)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🚀 Quickstart

### 💻 Installation

> AgentScope requires **Python 3.10** or higher.

#### 🛠️ From Source

```bash
# Clone the repository
git clone -b main https://github.com/agentscope-ai/agentscope.git

# Navigate to the project directory
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

A simple "Hello World" example to get you started:

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

Real-time interruption and control are supported by ReactAgent, allowing your agents to be more reactive:

<p align="center">
  <img src="./assets/images/realtime_steering_zh.gif" alt="Realtime Steering" width="49%"/>
  <img src="./assets/images/realtime_steering_en.gif" alt="Realtime Steering" width="49%"/>
</p>

### 🛠️ Fine-Grained MCP Control

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

AgentScope provides features to build and deploy multi-agent workflows.

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

Trace and visualize your agent applications with AgentScope Studio:

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

Comprehensive documentation is available to guide you through AgentScope's features and capabilities:

*   **Tutorial:**  Comprehensive guides covering:
    *   [Installation](https://doc.agentscope.io/tutorial/quickstart_installation.html)
    *   [Key Concepts](https://doc.agentscope.io/tutorial/quickstart_key_concept.html)
    *   [Create Message](https://doc.agentscope.io/tutorial/quickstart_message.html)
    *   [ReAct Agent](https://doc.agentscope.io/tutorial/quickstart_agent.html)
*   **Workflow:** Learn to build complex workflows:
    *   [Conversation](https://doc.agentscope.io/tutorial/workflow_conversation.html)
    *   [Multi-Agent Debate](https://doc.agentscope.io/tutorial/workflow_multiagent_debate.html)
    *   [Concurrent Agents](https://doc.agentscope.io/tutorial/workflow_concurrent_agents.html)
    *   [Routing](https://doc.agentscope.io/tutorial/workflow_routing.html)
    *   [Handoffs](https://doc.agentscope.io/tutorial/workflow_handoffs.html)
*   **FAQ:**  Get answers to common questions:
    *   [FAQ](https://doc.agentscope.io/tutorial/faq.html)
*   **Task Guides:** Deep dives into core modules:
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
*   **API:** Detailed API documentation:
    *   [API Docs](https://doc.agentscope.io/api/agentscope.html)
*   **Examples:** Practical code examples:
    *   [Examples](https://github.com/agentscope-ai/agentscope/tree/main/examples)
        *   Game, Workflow, Evaluation, and Functional examples.

## ⚖️ License

AgentScope is licensed under the Apache License 2.0.

## 📚 Publications

If you use AgentScope in your research, please cite our papers:

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

Thank you to all our contributors!

<a href="https://github.com/agentscope-ai/agentscope/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=agentscope-ai/agentscope&max=999&columns=12&anon=1" />
</a>