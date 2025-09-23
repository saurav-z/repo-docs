# AgentScope: Build Powerful LLM Applications with Agent-Oriented Programming

**AgentScope empowers developers to build robust and customizable LLM applications through agent-oriented programming, enabling innovative solutions with ease.**  Explore the power of [AgentScope on GitHub](https://github.com/agentscope-ai/agentscope).

[![arXiv](https://img.shields.io/badge/cs.MA-2402.14034-B31C1C?logo=arxiv&logoColor=B31C1C)](https://arxiv.org/abs/2402.14034)
[![PyPI](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://pypi.org/project/agentscope/)
[![PyPI Version](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fagentscope%2Fjson&query=%24.info.version&prefix=v&logo=pypi&label=version)](https://pypi.org/project/agentscope/)
[![Documentation](https://img.shields.io/badge/Docs-English%7C%E4%B8%AD%E6%96%87-blue?logo=markdown)](https://doc.agentscope.io/)
[![AgentScope Studio](https://img.shields.io/badge/GUI-AgentScope_Studio-blue?logo=look&logoColor=green&color=dark-green)](https://agentscope.io/)
[![License](https://img.shields.io/badge/license-Apache--2.0-black)](./LICENSE)

<p align="center">
<img src="https://trendshift.io/api/badge/repositories/10079" alt="modelscope%2Fagentscope | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</p>

## Key Features of AgentScope

*   **Transparent Development:**  Everything is visible and controllable. Prompt engineering, API calls, agent building, and workflow orchestration are all transparent.
*   **Realtime Steering:** Native support for real-time interruption and customized handling.
*   **Agentic Capabilities:**  Supports agentic tools management, long-term memory control, and RAG (Retrieval-Augmented Generation).
*   **Model Agnostic:**  Write your code once, run it with various LLM models.
*   **Modular Design:**  LEGO-style agent building with independent, modular components.
*   **Multi-Agent Focused:** Designed for multi-agent interactions, explicit message passing, and flexible workflow orchestration.
*   **Highly Customizable:**  Extensive customization options for tools, prompts, agents, workflows, and integrations.

<p align="center">
<img src="./assets/images/agentscope_v1_0822.png" alt="AgentScope Framework" width="80%"/>
</p>

## Why Choose AgentScope?

AgentScope provides a powerful and flexible framework, designed to be easy to learn for beginners while offering advanced features for experienced developers.  AgentScope lets you build and deploy AI agents and multi-agent applications with ease and transparency.

## Recent Updates

*   **[2024-09]** **Voice agent** is online! `ReActAgent` supports Qwen-Omni and GPT-Audio natively now, check our [new example](https://github.com/agentscope-ai/agentscope/tree/main/examples/agent/voice_agent) and [roadmap](https://github.com/agentscope-ai/agentscope/issues/773).
*   **[2024-09]** A new powerful 📋**Plan** module is online now! Check out the [tutorial](https://doc.agentscope.io/tutorial/task_plan.html) for more details.
*   **[2024-09]** **AgentScope Runtime** is open-sourced now! Enabling effective agent deployment with sandboxed tool execution for production-ready AI applications. Check out the [GitHub repo](https://github.com/agentscope-ai/agentscope-runtime).
*   **[2024-09]** **AgentScope Studio** is open-sourced now! Check out the [GitHub repo](https://github.com/agentscope-ai/agentscope-studio).
*   **[2024-08]** The new tutorial of v1 is online now! Check out the [tutorial](https://doc.agentscope.io) for more details.
*   **[2024-08]** 🎉🎉 AgentScope v1 is released now! This version fully embraces the asynchronous execution, providing many new features and improvements. Check out [changelog](https://github.com/agentscope-ai/agentscope/blob/main/docs/changelog.md) for detailed changes.

## Quickstart

### Installation

*   Requires Python 3.10 or higher.

**From Source:**

```bash
git clone -b main https://github.com/agentscope-ai/agentscope.git
cd agentscope
pip install -e .
```

**From PyPi:**

```bash
pip install agentscope
```

### Example

#### Hello AgentScope!

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

Natively support **realtime interruption** in ``ReActAgent`` with robust memory preservation, and convert interruption into an **observable event** for agent to seamlessly resume conversations.

<p align="center">
  <img src="./assets/images/realtime_steering_zh.gif" alt="Realtime Steering" width="49%"/>
  <img src="./assets/images/realtime_steering_en.gif" alt="Realtime Steering" width="49%"/>
</p>

### Fine-Grained MCP Control

Developers can obtain the MCP tool as a **local callable function**, and use it anywhere (e.g. call directly, pass to agent, wrap into a more complex tool, etc.)

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

### Multi-Agent Conversation

AgentScope provides ``MsgHub`` and pipelines to streamline multi-agent conversations, offering efficient message routing and seamless information sharing

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

Use the following command to install and start AgentScope Studio, to trace and visualize your agent application.

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

Comprehensive documentation is available to guide you through the framework:

*   [Tutorial](https://doc.agentscope.io/)
*   [API Reference](https://doc.agentscope.io/api/agentscope.html)
*   [Examples](https://github.com/agentscope-ai/agentscope/tree/main/examples)

## License

AgentScope is released under the Apache License 2.0.

## Publications

Cite our papers:

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

Thank you to all our contributors!

<a href="https://github.com/agentscope-ai/agentscope/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=agentscope-ai/agentscope&max=999&columns=12&anon=1" />
</a>
```
Key improvements and SEO considerations:

*   **Concise Hook:** A one-sentence hook to immediately grab attention.
*   **Keyword Optimization:** Included keywords like "LLM applications," "agent-oriented programming," "AI agents," and "multi-agent."
*   **Clear Headings:** Use of H2 and H3 tags for better organization and SEO.
*   **Bulleted Key Features:**  Easy-to-read bullet points highlight core functionality.
*   **Direct Links:**  Links back to the repository are prominent.
*   **Call to Actions:** Uses phrases like "Explore the power of..." and "Why Choose..." to encourage engagement.
*   **Complete Quickstart:**  Kept the quickstart installation and example.
*   **Clean Organization:** Improved readability and flow.
*   **Updated Timeline:** Added sections that highlight the recent updates and how the features can be implemented.
*   **Added Subheadings for Examples:** Added subheadings for the various examples.