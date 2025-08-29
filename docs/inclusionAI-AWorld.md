<!-- The following Markdown is SEO-optimized and incorporates key features and a concise hook. -->

# AWorld: Unlock the Future of AI with Self-Improving Agents

**AWorld empowers you to build AI agents that learn and evolve, paving the way for the next generation of intelligent systems.** Explore the power of agent self-improvement and collective intelligence!

[Link to Original Repo: https://github.com/inclusionAI/AWorld](https://github.com/inclusionAI/AWorld)

---

## Key Features

*   **🚀 Self-Improvement:** Enables agents to continuously refine their knowledge, skills, and performance.
*   **🤖 Multi-Agent Systems (MAS):** Facilitates the creation of complex, interacting agent societies using plug-and-play protocols.
*   **☁️ Cloud-Native Scalability:** Provides the infrastructure to train smarter agents and scale your projects.
*   **🧠 Collective Intelligence:** Fosters collaboration and knowledge-sharing to achieve advanced results across diverse domains.
*   **🧩 Flexible & Adaptable:** Quickly build individual tool-using agents or orchestrate sophisticated multi-agent systems.

---

## Collective Intelligence Achievements 🏆

AWorld is demonstrating cutting-edge capabilities in various areas:

| Category  | Achievement                                          | Performance                                                              | Key Innovation                       | Date       |
| :-------- | :--------------------------------------------------- | :----------------------------------------------------------------------- | :----------------------------------- | :--------- |
| 🤖 Agent  | GAIA Benchmark Excellence                         | Pass@1: **67.89**, Pass@3: **83.49** (109 tasks)                        | Multi-agent system stability & orchestration | 2025/08/06 |
| 🧠 Reasoning | IMO 2025 Problem Solving                        | **5/6** problems solved in 6 hours                                      | Multi-agent collaboration beats solo models | 2025/07/25 |

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

Get started with AWorld by following these simple steps:

### Prerequisites

```
Python>=3.11
```

### Installation

```bash
git clone https://github.com/inclusionAI/AWorld && cd AWorld
python setup.py install
```

### Hello World Examples

**Single Agent:**

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

**Multi-Agent (Swarm):**

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

### Running Your Agents

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

---

## 🏗️ Architecture Design Principles

AWorld is designed to facilitate collaboration, self-improvement, and the construction of advanced agent systems through a modular and adaptable architecture.

### Concepts & Framework

| Concept             | Description                                                                                                                                                         |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `agent`             | Defines the core classes, descriptions, output parsing, and multi-agent collaboration logic.                                                                           |
| `runner`            | Manages the execution loop for agents, handling episode rollouts, and parallel training workflows.                                                                      |
| `task`              | Encapsulates environment objectives, necessary tools, and termination conditions for agent interactions.                                                               |
| `swarm`             | Manages multi-agent coordination and emergent group behaviors through decentralized policies.                                                                          |
| `sandbox`           | Provides a controlled runtime for rapid prototyping and validation of agent behaviors.                                                                                   |
| `tools`             | Defines, adapts, and executes tools for agent-environment interaction.                                                                                                  |
| `context`           | Enables state tracking, configuration management, prompt optimization, multi-task state handling, and dynamic prompt templating.                                           |
| `memory`            | Supports short-term and long-term memory, summarization, retrieval, embeddings, and integration with vector databases.                                                      |
| `trace`             | Provides distributed tracing, context propagation, span management, and integration with monitoring tools for analyzing agent execution.                                     |

> 💡 Explore diverse AWorld applications in the [examples](./examples/) directory.

### Characteristics

| Agent Construction         | Topology Orchestration      | Environment                    |
|:---------------------------|:----------------------------|:-------------------------------|
| ✅ Integrated MCP services | ✅ Encapsulated runtime  | ✅ Runtime state management  |
| ✅ Multi-model providers   | ✅ Flexible MAS patterns | ✅ High-concurrency support  |
| ✅ Customization options   | ✅ Clear state tracing   | ✅ Distributed training      |

---

## 🧩 Applications

AWorld empowers you to construct robust agents and multi-agent systems for various applications:

### Multi-Agent Systems for Model Evolutions

By constructing diverse runtime environments, AWorld aims to push the boundaries of models and continuously advance intelligence.

| Category        | Runtime                          | Performance                                                                         | Key Information                    |
| --------------- | -------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------- |
| **Tool Use**    | Function call runtime construction [`tutorial`][funreason-model-url] | Competitive on BFCL benchmark  <br> ![Agent Framework](readme_assets/bfclv2_leaderboard.png) | ![Dataset][huggingface-dataset-image] <br> [![Model][huggingface-model-image]][funreason-model-url] <br> [![Paper][arxiv-image]][funreason-paper-url] <br> ![Blog][blog-image] <br> [![Code][github-code-image]][funreason-code-url] |
| **Deep Search** | Search runtime to be released      | SOTA on HotpotQA benchmark  <br> ![Agent Framework](readme_assets/hotpotqa_benchmark.png) | [![Dataset][huggingface-dataset-image]][deepsearch-dataset-url] <br> [![Model][huggingface-model-image]][deepsearch-model-url] <br> [![Paper][arxiv-image]][deepsearch-paper-url] <br> [![Code][github-code-image]][deepsearch-code-url]      |

### Multi-Agent Systems for Applications

AWorld's plug-and-play MAS architecture enables the rapid development of real-world web applications:

*   **Code generation & execution**
*   **Browser automation & tool use**
*   **Multimodal understanding & generation**

See [Appendix: Web Client Usage](#appendix-web-client-usage) for GAIA implementation examples.

---

## Contributing

Join us in building and improving AWorld! We welcome contributions from developers of all levels. Your help is vital to the project's success.

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

---

## Appendix: Web Client Usage

![GAIA Agent Runtime Demo](readme_assets/gaia_demo.gif)

Follow these steps to deploy and use your agents with the web client:

### Project Structure

```text
agent-project-root-dir/
    agent_deploy/
      my_first_agent/
        __init__.py
        agent.py
```

### Step 1: Define Your Agent

**`agent_deploy/my_first_agent/__init__.py`:** (Create an empty file)

```shell
cd agent_deploy/my_first_agent
touch __init__.py
```

**`agent_deploy/my_first_agent/agent.py`:** Implement your agent logic:

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

### Step 2: Run Agent

**Set environment variables:**

```shell
cd ${agent-project-root-dir}

export LLM_MODEL_NAME="gpt-4"
export LLM_API_KEY="your-api-key-here"
export LLM_BASE_URL="https://api.openai.com/v1"  # Optional for OpenAI
```

**Launch Your Agent:**

```shell
# Option 1: Launch with Web UI
aworld web
# Then open http://localhost:8000 in your browser

# Option 2: Launch REST API (For integrations)
aworld api
# Then visit http://localhost:8000/docs for API documentation
```

**Congratulations!** Your agent is now running and ready to chat!

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
<!-- resource section end -->