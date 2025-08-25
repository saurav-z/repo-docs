<!-- omit in the summary -->
<div align="center">

# AWorld: The Agent Runtime for Self-Improvement

</div>

<h4 align="center">

*"Self-awareness: the hardest problem isn't solving within limits, it's discovering one's own limitations"*

[![Twitter Follow][twitter-image]][twitter-url]
[![WeChat QR Code][wechat-image]][wechat-url]
[![Discord][discord-image]][discord-url]
[![License: MIT][license-image]][license-url]
[![DeepWiki][deepwiki-image]][deepwiki-url]
<!-- [![arXiv][arxiv-image]][arxiv-url] -->

</h4>

<h4 align="center">

[中文版](./README_zh.md) |
[Quickstart](#️-quickstart) |
[Architecture](#️-architecture-design-principles) |
[Applications](#-applications) |
[Contributing](#contributing) |
[Appendix](#appendix-web-client-usage)

</h4>

---
<!-- **AWorld (Agent World)** is a next-generation framework for agent learning with three key characteristics: 
1. **Plug-and-Play:** Box up complex modules with bulletproof protocols and zero-drama state control.
2. **Cloud-Native Velocity:** Train smarter agents that evolve their own brains—prompts, workflows, memory, and tools—on the fly.  
3. **Self-Awareness**: Synthesize the agent's own knowledge and experience to achieve ultimate self-improvement. -->

![](./readme_assets/heading_banner.png)

## AWorld: Powering Self-Improving AI Agents

**AWorld is a cutting-edge framework that enables the creation of self-improving AI agents by leveraging multi-agent systems, cloud-native scalability, and the synthesis of knowledge and experience.** This allows AI agents to continuously evolve and enhance their capabilities.

**Key Features:**

*   **Multi-Agent Systems (MAS):** Build complex, interacting agent societies using plug-and-play protocols and robust context management.
*   **Intelligence Beyond a Single Model:** Generate high-quality feedback and diverse synthetic training data that fuel individual agent evolution.
*   **Cloud-Native for Diversity & Scale:** Deliver the high concurrency and scalability for training smarter agents and achieving self-improvement.

AWorld empowers you to build tool-using agents, orchestrate sophisticated multi-agent systems, train agents effectively, and synthesize the high-quality data required for continuous agent evolution – all converging towards autonomous self-improvement.

---

**Collective Intelligence Achievements 🚀**

Demonstrating collective intelligence across diverse domains. Join us in the ongoing projects!

<table style="width: 100%; border-collapse: collapse; table-layout: fixed;">
  <thead>
    <tr>
      <th style="width: 30%; text-align: left; border-bottom: 2px solid #ddd; padding: 8px;">Category</th>
      <th style="width: 20%; text-align: left; border-bottom: 2px solid #ddd; padding: 8px;">Achievement</th>
      <th style="width: 20%; text-align: left; border-bottom: 2px solid #ddd; padding: 8px;">Performance</th>
      <th style="width: 25%; text-align: left; border-bottom: 2px solid #ddd; padding: 8px;">Key Innovation</th>
      <th style="width: 5%; text-align: left; border-bottom: 2px solid #ddd; padding: 8px;">Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; vertical-align: top;">🤖 Agent</td>
      <td style="padding: 8px; vertical-align: top;">
        <strong>GAIA Benchmark <br>Excellence</strong>
        <br>
        <a href="https://huggingface.co/spaces/gaia-benchmark/leaderboard" target="_blank" style="text-decoration: none;">
          <img src="https://img.shields.io/badge/GAIA-Leaderboard-blue" alt="GAIA">
        </a>
      </td>
      <td style="padding: 8px; vertical-align: top;">
        Pass@1: <strong>67.89</strong> <br>
        Pass@3: <strong>83.49</strong>
        <br> (109 tasks)
        <a href="./examples/gaia/README_GUARD.md" target="_blank" style="text-decoration: none;">
          <img src="https://img.shields.io/badge/Code-README-green" alt="Code">
        </a>
      </td>
      <td style="padding: 8px; vertical-align: top;">
        Multi-agent system <br>stability & orchestration
        <br>
        <a href="https://arxiv.org/abs/2508.09889" target="_blank" style="text-decoration: none;">
          <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper">
        </a>
      </td>
      <td style="padding: 8px; vertical-align: top;">2025/08/06</td>
    </tr>
    <tr>
      <td style="padding: 8px; vertical-align: top;">🧠 Reasoning</td>
      <td style="padding: 8px; vertical-align: top;">
        <strong>IMO 2025 <br>Problem Solving</strong>
        <br>
        <a href="https://www.imo-official.org/year_info.aspx?year=2025" target="_blank" style="text-decoration: none;">
          <img src="https://img.shields.io/badge/IMO-2025-blue" alt="IMO">
        </a>
      </td>
      <td style="padding: 8px; vertical-align: top;">
        <strong>5/6</strong> problems <br>solved in 6 hours
        <br>
        <a href="examples/imo/README.md" target="_blank" style="text-decoration: none;">
          <img src="https://img.shields.io/badge/Code-README-green" alt="Code">
        </a>
      </td>
      <td style="padding: 8px; vertical-align: top;">Multi-agent collaboration <br>beats solo models</td>
      <td style="padding: 8px; vertical-align: top;">2025/07/25</td>
    </tr>
  </tbody>
</table>

<details>
<summary style="font-size: 1.2em;font-weight: bold;"> 🌏 View ongoing projects </summary>

<table style="width: 100%; border-collapse: collapse; table-layout: fixed;">
  <thead>
    <tr>
      <th style="width: 20%; text-align: left; border-bottom: 2px solid #ddd; padding: 8px;">Category</th>
      <th style="width: 35%; text-align: left; border-bottom: 2px solid #ddd; padding: 8px;">Achievement</th>
      <th style="width: 10%; text-align: left; border-bottom: 2px solid #ddd; padding: 8px;">Status</th>
      <th style="width: 35%; text-align: left; border-bottom: 2px solid #ddd; padding: 8px;">Expected Impact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; vertical-align: top;">🖼️ Multi-Modal</td>
      <td style="padding: 8px; vertical-align: top;">Advanced OS / Web Interaction</td>
      <td style="padding: 8px; vertical-align: top;">In Progress</td>
      <td style="padding: 8px; vertical-align: top;">Visual reasoning <br>environment understanding</td>
    </tr>
    <tr>
      <td style="padding: 8px; vertical-align: top;">💻 Code</td>
      <td style="padding: 8px; vertical-align: top;">Advanced installation, coding, <br>testing, debugging, etc. ability</td>
      <td style="padding: 8px; vertical-align: top;">In Progress</td>
      <td style="padding: 8px; vertical-align: top;">Automated software <br>engineering capabilities</td>
    </tr>
    <tr>
      <td style="padding: 8px; vertical-align: top;">🔧 Tool Use</td>
      <td style="padding: 8px; vertical-align: top;">Advanced multi-turn function call</td>
      <td style="padding: 8px; vertical-align: top;">Coming soon</td>
      <td style="padding: 8px; vertical-align: top;">Impact the real world</td>
    </tr>
  </tbody>
</table>

</details>

---

## 🏃‍♀️ Quickstart

A step-by-step guide to get you started with AWorld.

### Prerequisites

> [!TIP]
> Python>=3.11

```bash
git clone https://github.com/inclusionAI/AWorld && cd AWorld
python setup.py install
```

### Hello World Examples

#### Single Agent

```python
import os

from aworld.agents.llm_agent import Agent
from aworld.runner import Runners

summarizer = Agent(
    name="Summary Agent", 
    system_prompt="You specialize at summarizing.",
)

result = Runners.sync_run(
    input="Tell me a succint history about the universe", 
    agent=summarizer,
)
```

#### Multi-Agent (Swarm)

```python
import os

from aworld.agents.llm_agent import Agent
from aworld.runner import Runners
from aworld.core.agent.swarm import Swarm

researcher = Agent(
    name="Research Agent", 
    system_prompt="You specialize at researching.",
)
summarizer = Agent(
    name="Summary Agent", 
    system_prompt="You specialize at summarizing.",
)
# Create agent team with collaborative workflow
team = Swarm(researcher, summarizer)

result = Runners.sync_run(
    input="Tell me a complete history about the universe", 
    swarm=team,
)
```

#### Run Agents

```bash
# Set LLM credentials
export LLM_MODEL_NAME="gpt-4"
export LLM_API_KEY="your-api-key-here"
export LLM_BASE_URL="https://api.openai.com/v1"

# Run
python /path/to/agents/or/teams
```

<details>
<summary style="font-size: 1.2em;font-weight: bold;"> 🌏 Click to View Advanced Usages </summary>

### Pass AgentConfig Explicitly
```python
import os

from aworld.agents.llm_agent import Agent
from aworld.runner import Runners
from aworld.config.conf import AgentConfig
from aworld.core.agent.swarm import Swarm

gpt_conf = AgentConfig(
    llm_provider="openai",
    llm_model_name="gpt-4o",
    llm_api_key="<OPENAI_API_KEY>",
    llm_temperature=0.1,
)
openrouter_conf = AgentConfig(
    llm_provider="openai",
    llm_model_name="google/gemini-2.5-pro",
    llm_api_key="<OPENROUTER_API_KEY>",
    llm_base_url="https://openrouter.ai/api/v1"
    llm_temperature=0.1,
)

researcher = Agent(
    name="Research Agent", 
    conf=gpt_conf,
    system_prompt="You specialize at researching.",
)
summarizer = Agent(
    name="Summary Agent", 
    conf=openrouter_conf,
    system_prompt="You specialize at summarizing.",
)
# Create agent team with collaborative workflow
team = Swarm(researcher, summarizer)

result = Runners.sync_run(
    input="Tell me a complete history about the universe", 
    swarm=team,
)
```

### Agent Equipped with MCP Tools
```python
import os

from aworld.agents.llm_agent import Agent
from aworld.runner import Runners

mcp_config = {
    "mcpServers": {
        "GorillaFileSystem": {
            "type": "stdio",
            "command": "python",
            "args": ["examples/BFCL/mcp_tools/gorilla_file_system.py"],
        },
    }
}

file_sys = Agent(
    name="file_sys_agent",
    system_prompt=(
        "You are a helpful agent to use "
        "the standard file system to perform file operations."
    ),
    mcp_servers=mcp_config.get("mcpServers", []).keys(),
    mcp_config=mcp_config,
)

result = Runners.sync_run(
    input=(
        "use mcp tools in the GorillaFileSystem server "
        "to perform file operations: "
        "write the content 'AWorld' into "
        "the hello_world.py file with a new line "
        "and keep the original content of the file. "
        "Make sure the new and old "
        "content are all in the file; "
        "and display the content of the file"
    ),
    agent=file_sys,
)
```

### Agent Integrated with Memory
It is recommended to use `MemoryFactory` to initialize and access Memory instances.

```python
from aworld.memory.main import MemoryFactory
from aworld.core.memory import MemoryConfig, MemoryLLMConfig

# Simple initialization
memory = MemoryFactory.instance()

# Initialization with LLM configuration
MemoryFactory.init(
    config=MemoryConfig(
        provider="aworld",
        llm_config=MemoryLLMConfig(
            provider="openai",
            model_name=os.environ["LLM_MODEL_NAME"],
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"]
        )
    )
)
memory = MemoryFactory.instance()
```

`MemoryConfig` allows you to integrate different embedding models and vector databases.
```python
import os

from aworld.core.memory import MemoryConfig, MemoryLLMConfig, EmbeddingsConfig, VectorDBConfig

MemoryFactory.init(
    config=MemoryConfig(
        provider="aworld",
        llm_config=MemoryLLMConfig(
            provider="openai",
            model_name=os.environ["LLM_MODEL_NAME"],
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"]
        ),
        embedding_config=EmbeddingsConfig(
            provider="ollama", # or huggingface, openai, etc.
            base_url="http://localhost:11434",
            model_name="nomic-embed-text"
        ),
        vector_store_config=VectorDBConfig(
            provider="chroma",
            config={
                "chroma_data_path": "./chroma_db",
                "collection_name": "aworld",
            }
        )
    )
)
```

### Mutil-Agent Systems
We present a classic topology: `Leader-Executor`.
```python
"""
Leader-Executor topology:
 ┌───── plan ───┐     
exec1         exec2

Each agent communicates with a single supervisor agent, 
well recognized as Leader-Executor topology, 
also referred to as a team topology in Aworld.
"""
from aworld.agents.llm_agent import Agent
from aworld.core.agent.swarm import TeamSwarm

plan = Agent(name="plan", conf=agent_conf)
exec1 = Agent(name="exec1", conf=agent_conf)
exec2 = Agent(name="exec2", conf=agent_conf)
swarm = TeamSwarm(plan, exec1, exec2)
```
Optionally, you can use `Handsoff` mechanism to customize your own topology.
```python
from aworld.core.agent.swarm import HandoffSwarm
swarm = HandoffSwarm((plan, exec1), (plan, exec2))
```

</details>

## 🏗️ Architecture Design Principles

AWorld provides a comprehensive environment that supports a diverse array of applications, such as `Product Prototype Verification`, `Foundational Model Training`, and the design of `Multi-Agent Systems (MAS)` through meta-learning. 

This framework is engineered to be highly adaptable, enabling researchers and developers to explore and innovate across multiple domains, thereby advancing the capabilities and applications of multi-agent systems.

### Concepts & Framework

| Concepts | Description |
| :-------------------------------------- | ------------ |
| [`agent`](./aworld/core/agent/base.py)  | Define the foundational classes, descriptions, output parsing, and multi-agent collaboration (swarm) logic for defining, managing, and orchestrating agents in the AWorld system. |
| [`runner`](./aworld/runners)            | Contains runner classes that manage the execution loop for agents in environments, handling episode rollouts and parallel training/evaluation workflows.   |
| [`task`](./aworld/core/task.py)         | Define the base Task class that encapsulates environment objectives, necessary tools, and termination conditions for agent interactions.  |
| [`swarm`](./aworld/core/agent/swarm.py) | Implement the SwarmAgent class managing multi-agent coordination and emergent group behaviors through decentralized policies. |
| [`sandbox`](./aworld/sandbox)           | Provide a controlled runtime with configurable scenarios for rapid prototyping and validation of agent behaviors. |
| [`tools`](./aworld/tools)               | Offer a flexible framework for defining, adapting, and executing tools for agent-environment interaction in the AWorld system. |
| [`context`](./aworld/core/context)      | Feature a comprehensive context management system for AWorld agents, enabling complete state tracking, configuration management, prompt optimization, multi-task state handling, and dynamic prompt templating throughout the agent lifecycle.  |
| [`memory`](./aworld/memory)             | Implement an extensible memory system for agents, supporting short-term and long-term memory, summarization, retrieval, embeddings, and integration.|
| [`trace`](./aworld/trace)               | Feature an observable tracing framework for AWorld, enabling distributed tracing, context propagation, span management, and integration with popular frameworks and protocols to monitor and analyze agent, tool, and task execution.|

> 💡 Check the [examples](./examples/) directory to explore diverse AWorld applications.

### Characteristics

| Agent Construction         | Topology Orchestration      | Environment                    |
|:---------------------------|:----------------------------|:-------------------------------|
| ✅ Integrated MCP services | ✅ Encapsulated runtime  | ✅ Runtime state management  |
| ✅ Multi-model providers   | ✅ Flexible MAS patterns | ✅ High-concurrency support  |
| ✅ Customization options   | ✅ Clear state tracing   | ✅ Distributed training      |

### Forward Process Design

![](readme_assets/runtime.jpg)

Here is a forward illustration to collect BFCL forward trajectories: [`tutorial`](./examples/BFCL/README.md).

### Backward Process Design

> During training, an action-state rollout demonstration using **AWorld's distributed environments**.

![](readme_assets/agent_training2.jpg)

> [!NOTE]
> An illustration of training code that seamlessly integrates the RL learning framework (Swift, in this example) with AWorld as the environment is shown below. This integration enables scalable and efficient agent training through distributed environment execution. (To run high-concurrency rollouts, you need to deploy an online distributed environment. Please contact [chenyi.zcy@antgroup.com](mailto:chenyi.zcy@antgroup.com) if assistance is needed.)

<details>
<summary style="font-size: 1.2em;font-weight: bold;"> 🌏 Click to View Tutorial Example</summary>
To apply and use this integration:

1. Clone AWorld's `agent_training_server` branch:
```bash
git clone -b agent_training_server --single-branch https://github.com/inclusionAI/AWorld.git
```

2. Clone ms-swift's v3.5.2 branch (shallow clone):
```bash
git clone -b v3.5.2 --depth=1 https://github.com/modelscope/ms-swift.git ms-swift
```

3. Copy patch files from AWorld to ms-swift:
```bash
cp -r AWorld/patches ms-swift/
```

4. Enter the patches directory and apply the patch:
```bash
cd ms-swift/patches
git apply 0001-feat-add-agent-training-support-with-aworld-server.patch
```
</details>

## 🧩 Applications

AWorld allows you to construct **agents** and **multi-agent systems** with ease. 

### Multi-Agent Systems for Model Evolutions

AWorld aims to reach the limitations of models and continuously push intelligence forward by constructing diverse runtime environments, such as tools, agents, and models.

The following is a list of successful proposal (with open-source models, technical reports, and code):

| Category | Runtime | <div style="width:400px">Performance</div> | <div style="width:100px;">Key Information</div> |
| --------------- | --------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------- |
| **Tool Use**    | Function call runtime construction [`tutorial`][funreason-model-url] | Competitive on BFCL benchmark  <br> ![Agent Framework](readme_assets/bfclv2_leaderboard.png) | ![Dataset][huggingface-dataset-image] <br> [![Model][huggingface-model-image]][funreason-model-url] <br> [![Paper][arxiv-image]][funreason-paper-url] <br> ![Blog][blog-image] <br> [![Code][github-code-image]][funreason-code-url] |
| **Deep Search** | Search runtime to be released           | SOTA on HotpotQA benchmark  <br> ![Agent Framework](readme_assets/hotpotqa_benchmark.png)    | [![Dataset][huggingface-dataset-image]][deepsearch-dataset-url] <br> [![Model][huggingface-model-image]][deepsearch-model-url] <br> [![Paper][arxiv-image]][deepsearch-paper-url] <br> [![Code][github-code-image]][deepsearch-code-url]      |

### Multi-Agent Systems for Applications

AWorld's plug-and-play MAS architecture enables **real-world web application development** beyond agent training.

Build production-ready systems that handle complex tasks through:

*   Code generation & execution
*   Browser automation & tool use
*   Multimodal understanding & generation
*   And many more to emerge!

See [Appendix: Web Client Usage](#appendix-web-client-usage) for GAIA implementation examples.

## Contributing

We warmly welcome developers to join us in building and improving AWorld! Whether you're interested in enhancing the framework, fixing bugs, or adding new features, your contributions are valuable to us.

For academic citations or wish to contact us, please use the following BibTeX entry:

```bibtex
@software{aworld2025,
  author = {Agent Team at InclusionAI},
  title = {AWorld: Enabling Agent Self-Improvement through Interactive Experience with Dynamic Runtime},
  year = {2025},
  url = {https://github.com/inclusionAI/AWorld},
  version = {0.1.0},
  publisher = {GitHub},
  email = {chenyi.zcy at antgroup.com}
}
```

## Star History
![](https://api.star-history.com/svg?repos=inclusionAI/AWorld&type=Date)

## Appendix: Web Client Usage
![GAIA Agent Runtime Demo](readme_assets/gaia_demo.gif)

Your project structure should look like this:
```text
agent-project-root-dir/
    agent_deploy/
      my_first_agent/
        __init__.py
        agent.py
```

Create project folders.

```shell
mkdir my-aworld-project && cd my-aworld-project # project-root-dir
mkdir -p agent_deploy/my_first_agent
```

#### Step 1: Define Your Agent

Create your first agnet in `agent_deploy/my_first_agent`:

`__init__.py`: Create empty `__ini__.py` file.

```shell
cd agent_deploy/my_first_agent
touch __init__.py
```

`agent.py`: Define your agent logic:

```python
import logging
import os
from aworld.cmd.data_model import BaseAWorldAgent, ChatCompletionRequest
from aworld.config.conf import AgentConfig, TaskConfig
from aworld.agents.llm_agent import Agent
from aworld.core.task import Task
from aworld.runner import Runners

logger = logging.getLogger(__name__)

class AWorldAgent(BaseAWorldAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self):
        return "My First Agent"

    def description(self):
        return "A helpful assistant that can answer questions and help with tasks"

    async def run(self, prompt: str = None, request: ChatCompletionRequest = None):
        # Load LLM configuration from environment variables
        agent_config = AgentConfig(
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
        )

        # Validate required configuration
        if not agent_config.llm_model_name or not agent_config.llm_api_key:
            raise ValueError("LLM_MODEL_NAME and LLM_API_KEY must be set!")

        # Optional: Configure MCP tools for enhanced capabilities
        mcp_config = {
            "mcpServers": {
                "amap-mcp": {
                    "type": "sse",
                    "url": "https://mcp.example.com/sse?key=YOUR_API_KEY", # Replace Your API Key
                    "timeout": 30,
                    "sse_read_timeout": 300
                }
            }
        }

        # Create the agent instance
        agent = Agent(
            conf=agent_config,
            name="My First Agent",
            system_prompt="""You are a helpful AI assistant. Your goal is to:
            - Answer questions accurately and helpfully
            - Provide clear, step-by-step guidance when needed
            - Be friendly and professional in your responses""",
            mcp_servers=["amap-mcp"],
            mcp_config=mcp_config
        )

        # Extract user input
        user_input = prompt or (request.messages[-1].content if request else "")
        
        # Create and execute task
        task = Task(
            input=user_input,
            agent=agent,
            conf=TaskConfig(max_steps=5),
            session_id=getattr(request, 'session_id', None)
        )

        # Stream the agent's response
        async for output in Runners.streamed_run_task(task).stream_events():
            yield output
```

#### Step 2: Run Agent

Setup environment variables:

```shell
# Navigate back to project root
cd ${agent-project-root-dir}

# Set your LLM credentials
export LLM_MODEL_NAME="gpt-4"
export LLM_API_KEY="your-api-key-here"
export LLM_BASE_URL="https://api.openai.com/v1"  # Optional for OpenAI
```

Launch Your Agent:
```shell
# Option 1: Launch with Web UI
aworld web
# Then open http://localhost:8000 in your browser

# Option 2: Launch REST API (For integrations)
aworld api
# Then visit http://localhost:8000/docs for API documentation
```

Success! Your agent is now running and ready to chat!

---
<!-- resource section start -->
<!-- image links -->
[arxiv-image]: https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white
[blog-image]: https://img.shields.io/badge/Blog-Coming%20Soon-FF5722?style=for-the-badge&logo=blogger&logoColor=white
[deepwiki-image]: https://img.shields.io/badge/DeepWiki-Explore-blueviolet?logo=wikipedia&logoColor=white
[discord-image]: https://img.shields.io/badge/Discord-Join%20us-blue?logo=discord&logoColor=white
[github-code-image]: https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white
[huggingface-dataset-image]: https://img.shields.io/badge/Dataset-Coming%20Soon-007ACC?style=for-the-badge&logo=dataset&logoColor=white
[huggingface-model-image]: https://img.shields.io/badge/Model-Hugging%20Face-FF6B6B?style=for-the-badge&logo=huggingface&logoColor=white
[license-image]: https://img.shields.io/badge/License-MIT-yellow.svg
[twitter-image]: https://img.shields.io/twitter/follow/AWorld_AI?style=social
[wechat-image]: https://img.shields.io/badge/WeChat-Add%20us-green?logo=wechat&logoColor=white

<!-- aworld links -->
[deepwiki-url]: https://deepwiki.com/inclusionAI/AWorld
[discord-url]: https://discord.gg/b4Asj2ynMw
[license-url]: https://opensource.org/licenses/MIT
[twitter-url]: https://x.com/InclusionAI666
[wechat-url]: https://raw.githubusercontent.com/inclusionAI/AWorld/main/readme_assets/aworld_wechat.png

<!-- funreason links -->
[funreason-code-url]: https://github.com/BingguangHao/FunReason
[funreason-model-url]: https://huggingface.co/Bingguang/FunReason
[funreason-paper-url]: https://arxiv.org/pdf/2505.20192
<!-- [funreason-dataset-url]: https://github.com/BingguangHao/FunReason -->
<!-- [funreason-blog-url]: https://github.com/BingguangHao/FunReason -->

<!-- deepsearch links -->
[deepsearch-code-url]: https://github.com/inclusionAI/AgenticLearning
[deepsearch-dataset-url]: https://github.com/inclusionAI/AgenticLearning
[deepsearch-model-url]: https://huggingface.co/collections/endertzw/rag-r1-68481d7694b3fca8b809aa29
[deepsearch-paper-url]: https://arxiv.org/abs/2507.02962

<!-- badge -->
[MAS]: https://img.shields.io/badge/Mutli--Agent-System-EEE1CE
[IMO]: https://img.shields.io/badge/IMO-299D8F
[BFCL]: https://img.shields.io/badge/BFCL-8AB07D
[GAIA]: https://img.shields.io/badge/GAIA-E66F51
[Runtime]: https://img.shields.io/badge/AWorld-Runtime-287271
[Leaderboard]: https://img.shields.io/badge/Leaderboard-FFE6B7
[Benchmark]: https://img.shields.io/badge/Benchmark-FFE6B7
[Cloud-Native]: https://img.shields.io/badge/Cloud--Native-B19CD7
[Forward]: https://img.shields.io/badge/Forward-4A90E2
[Backward]: https://img.shields.io/badge/Backward-7B68EE
[Code]: https://img.shields.io/badge/Code-FF6B6B
[Paper]: https://img.shields.io/badge/Paper-4ECDC4