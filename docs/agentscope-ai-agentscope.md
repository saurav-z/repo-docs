# AgentScope: Build Powerful LLM Applications with Agent-Oriented Programming

**AgentScope is a developer-friendly framework that simplifies the creation of sophisticated LLM applications using agent-oriented programming.** [Explore the AgentScope Repository](https://github.com/agentscope-ai/agentscope).

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

*   **Transparent Development:** AgentScope prioritizes transparency, giving developers full control over prompt engineering, API interactions, agent building, and workflow orchestration.
*   **Realtime Steering:** Native support for real-time interruption and customized handling, enhancing user interaction.
*   **Agentic Capabilities:** Supports agentic tools management, long-term memory control, and RAG (Retrieval-Augmented Generation) for more intelligent agents.
*   **Model Agnostic:** Write your code once and run it with various language models.
*   **Modular Design:** Build agents with independent, LEGO-style components for flexibility and reusability.
*   **Multi-Agent Focused:** Designed for multi-agent systems with explicit message passing and simplified workflow orchestration.
*   **Highly Customizable:**  Easily customize tools, prompts, agents, workflows, third-party libraries, and visualizations to meet your specific needs.

## AgentScope 1.0 Highlights

| Module      | Feature                                                                            | Tutorial                                                                |
|-------------|------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| model       | Async invocation, reasoning model support, streaming/non-streaming returns        | [Model](https://doc.agentscope.io/tutorial/task_model.html)             |
| tool        | Async/sync tool functions, streaming returns, user interruption, post-processing, group-wise & agentic tools management        | [Tool](https://doc.agentscope.io/tutorial/task_tool.html)               |
| MCP         | Streamable HTTP/SSE/StdIO transport, stateful/stateless mode, fine-grained control | [MCP](https://doc.agentscope.io/tutorial/task_mcp.html)                 |
| agent       | Async execution, parallel tool calls, realtime steering, state and long-term memory management, agent hooks                                                            |                                                                         |
| tracing     | OpenTelemetry-based tracing, third-party platform connections (Arize-Phoenix, Langfuse) | [Tracing](https://doc.agentscope.io/tutorial/task_tracing.html)         |
| memory      | Long-term memory support                                                           | [Memory](https://doc.agentscope.io/tutorial/task_long_term_memory.html) |
| session     | Session/application-level state management                                        | [Session](https://doc.agentscope.io/tutorial/task_state.html)           |
| evaluation  | Distributed and parallel evaluation                                        | [Evaluation](https://doc.agentscope.io/tutorial/task_eval.html)         |
| formatter   | Multi-agent prompt formatting with tools API, truncation-based strategy       | [Prompt Formatter](https://doc.agentscope.io/tutorial/task_prompt.html) |
| plan        | ReAct-based planning, manual plan specification                                             | [Plan](https://doc.agentscope.io/tutorial/task_plan.html)               |
| ...        |                                                                                    |                                                                         |

## What's New

*   **[2025-09]** Powerful 📋**Plan** module online!  [Tutorial](https://doc.agentscope.io/tutorial/task_plan.html).
*   **[2025-09]** **AgentScope Runtime** [GitHub repo](https://github.com/agentscope-ai/agentscope-runtime).
*   **[2025-09]** **AgentScope Studio** [GitHub repo](https://github.com/agentscope-ai/agentscope-studio).
*   **[2025-08]** v1 tutorial online!  [Tutorial](https://doc.agentscope.io).
*   **[2025-08]** 🎉🎉 AgentScope v1 Release! [changelog](https://github.com/agentscope-ai/agentscope/blob/main/docs/changelog.md)

## Quickstart

### Installation

*   **Requires:** Python 3.10+

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
    agent1 = ...
    agent2 = ...
    agent3 = ...
    agent4 = ...
    async with MsgHub(
        participants=[agent1, agent2, agent3],
        announcement=Msg("Host", "Introduce yourselves.", "assistant")
    ) as hub:
        await sequential_pipeline([agent1, agent2, agent3])
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
    *   [Installation](https://doc.agentscope.io/tutorial/quickstart_installation.html)
    *   [Key Concepts](https://doc.agentscope.io/tutorial/quickstart_key_concept.html)
    *   [Create Message](https://doc.agentscope.io/tutorial/quickstart_message.html)
    *   [ReAct Agent](https://doc.agentscope.io/tutorial/quickstart_agent.html)
*   Workflow
    *   [Conversation](https://doc.agentscope.io/tutorial/workflow_conversation.html)
    *   [Multi-Agent Debate](https://doc.agentscope.io/tutorial/workflow_multiagent_debate.html)
    *   [Concurrent Agents](https://doc.agentscope.io/tutorial/workflow_concurrent_agents.html)
    *   [Routing](https://doc.agentscope.io/tutorial/workflow_routing.html)
    *   [Handoffs](https://doc.agentscope.io/tutorial/workflow_handoffs.html)
*   FAQ
    *   [FAQ](https://doc.agentscope.io/tutorial/faq.html)
*   Task Guides
    *   [Model](https://doc.agentscope.io/tutorial/task_model.html)
    *   [Prompt Formatter](https://doc.agentscope.io/tutorial/task_prompt.html)
    *   [Tool](https://doc.agentscope.io/tutorial/task_tool.html)
    *   [Memory](https://doc.agentscope.io/tutorial/task_memory.html)
    *   [Long-Term Memory](https://doc.agentscope.io/tutorial/task_long_term_memory.html)
    *   [Agent](https://doc.agentscope.io/tutorial/task_agent.html)
    *   [Pipeline](https://doc.agentscope.io/tutorial/task_pipeline.html)
    *   [State/Session Management](https://doc.agentscope.io/tutorial/task_state.html)
    *   [Agent Hooks](https://doc.agentscope.io/tutorial/task_hook.html)
    *   [MCP](https://doc.agentscope.io/tutorial/task_mcp.html)
    *   [AgentScope Studio](https://doc.agentscope.io/tutorial/task_studio.html)
    *   [Tracing](https://doc.agentscope.io/tutorial/task_tracing.html)
    *   [Evaluation](https://doc.agentscope.io/tutorial/task_eval.html)
    *   [Embedding](https://doc.agentscope.io/tutorial/task_embedding.html)
    *   [Token](https://doc.agentscope.io/tutorial/task_token.html)
*   API
    *   [API Docs](https://doc.agentscope.io/api/agentscope.html)
*   [Examples](https://github.com/agentscope-ai/agentscope/tree/main/examples)

## License

AgentScope is released under the Apache License 2.0.

## Publications

If you find our work helpful for your research or application, please cite our papers.

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

## Contributors

Thanks to our contributors:

<a href="https://github.com/agentscope-ai/agentscope/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=agentscope-ai/agentscope&max=999&columns=12&anon=1" />
</a>