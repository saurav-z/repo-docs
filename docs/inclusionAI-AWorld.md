# AWorld: Revolutionizing Agent Self-Improvement 🚀

**AWorld is a cutting-edge framework designed for building intelligent agents that continuously learn and evolve.**  ([Original Repo](https://github.com/inclusionAI/AWorld))

---

## Key Features

*   **Plug-and-Play:** Simplify complex module integration with robust protocols and effortless state management.
*   **Cloud-Native Velocity:** Train agents that dynamically evolve their prompts, workflows, memory, and tools.
*   **Self-Awareness:** Empower agents to synthesize knowledge and experience for ultimate self-improvement.
*   **Build Workflows:** Design and implement automated task sequences.
*   **Create Intelligent Agents:** Build AI agents with MCP tools.
*   **Orchestrate Multi-Agent Systems (MAS):** Develop collaborative agent ecosystems.
*   **Optimize Agent Training:** Enhance performance within MAS across various environments.

---

## Collective Intelligence in Action 🚀

AWorld demonstrates collective intelligence across diverse domains.

| Category | Achievement                                                     | Performance                                                 | Key Innovation                                                | Date       |
| :------- | :-------------------------------------------------------------- | :---------------------------------------------------------- | :------------------------------------------------------------ | :--------- |
| 🤖 Agent | **GAIA Benchmark Excellence**                                    | Pass@1: **67.89**, Pass@3: **83.49** (109 tasks)              | Multi-agent system stability & orchestration                | 2025/08/06 |
| 🧠 Reasoning | **IMO 2025 Problem Solving**                                  | 5/6 problems solved in 6 hours                                | Multi-agent collaboration beats solo models                   | 2025/07/25 |

---

## Ongoing Projects 🌏

<details>
<summary style="font-size: 1.2em;font-weight: bold;"> Explore Current Initiatives </summary>

| Category        | Achievement                        | Status       | Expected Impact                                      |
| :-------------- | :--------------------------------- | :----------- | :--------------------------------------------------- |
| 🖼️ Multi-Modal  | Advanced OS / Web Interaction      | In Progress  | Visual reasoning & environment understanding        |
| 💻 Code         | Advanced coding and debugging skills | In Progress  | Automated software engineering capabilities          |
| 🔧 Tool Use     | Advanced multi-turn function call | Coming soon  | Impact the real world                               |

</details>

---

## Architecture & Design Principles

AWorld is a versatile multi-agent framework designed to facilitate collaborative interactions and self-improvement among agents. This framework is engineered to be highly adaptable, enabling researchers and developers to explore and innovate across multiple domains, thereby advancing the capabilities and applications of multi-agent systems.

### Core Concepts

*   **`agent`**: Defines foundational classes, descriptions, output parsing, and multi-agent collaboration (swarm) logic.
*   **`runner`**: Manages agent execution loops, episode rollouts, and parallel training/evaluation workflows.
*   **`task`**: Encapsulates environment objectives, tools, and termination conditions for agent interactions.
*   **`swarm`**: Manages multi-agent coordination and emergent group behaviors.
*   **`sandbox`**: Provides a controlled runtime for rapid prototyping.
*   **`tools`**: Offers a framework for defining and executing tools for agent-environment interaction.
*   **`context`**: Comprehensive context management system.
*   **`memory`**: Extensible memory system for agents, supporting short-term and long-term memory.
*   **`trace`**: Observability framework for monitoring and analyzing agent, tool, and task execution.

>   Explore diverse AWorld applications in the [examples](./examples/) directory.

### Key Characteristics

*   **Agent Construction:** Integrated MCP services, Multi-model providers, and Customization options
*   **Topology Orchestration:** Encapsulated runtime and Flexible MAS patterns
*   **Environment:** Runtime state management, High-concurrency support, Distributed training and Clear state tracing

### Architecture Overview

*   **Forward Process Design:** ([Tutorial](./examples/BFCL/README.md)).
*   **Backward Process Design:**  Agent training with AWorld's distributed environments.

<details>
<summary style="font-size: 1.2em;font-weight: bold;">  Tutorial Example </summary>

1.  Clone AWorld's `agent_training_server` branch:
    ```bash
    git clone -b agent_training_server --single-branch https://github.com/inclusionAI/AWorld.git
    ```

2.  Clone ms-swift's v3.5.2 branch (shallow clone):
    ```bash
    git clone -b v3.5.2 --depth=1 https://github.com/modelscope/ms-swift.git ms-swift
    ```

3.  Copy patch files from AWorld to ms-swift:
    ```bash
    cp -r AWorld/patches ms-swift/
    ```

4.  Enter the patches directory and apply the patch:
    ```bash
    cd ms-swift/patches
    git apply 0001-feat-add-agent-training-support-with-aworld-server.patch
    ```
</details>

---

## Applications

AWorld allows you to construct **agents** and **multi-agent systems** with ease. 

### Multi-Agent Systems for Model Evolutions

| Category       | Runtime                                            | Performance                                                                                  | Key Information                                                                                |
| :------------- | :------------------------------------------------- | :------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |
| **Tool Use**   | Function call runtime construction [`tutorial`][funreason-model-url] | Competitive on BFCL benchmark  <br> ![Agent Framework](readme_assets/bfclv2_leaderboard.png) | ![Dataset][huggingface-dataset-image] <br> [![Model][huggingface-model-image]][funreason-model-url] <br> [![Paper][arxiv-image]][funreason-paper-url] <br> ![Blog][blog-image] <br> [![Code][github-code-image]][funreason-code-url] |
| **Deep Search** | Search runtime to be released           | SOTA on HotpotQA benchmark  <br> ![Agent Framework](readme_assets/hotpotqa_benchmark.png)    | [![Dataset][huggingface-dataset-image]][deepsearch-dataset-url] <br> [![Model][huggingface-model-image]][deepsearch-model-url] <br> [![Paper][arxiv-image]][deepsearch-paper-url] <br> [![Code][github-code-image]][deepsearch-code-url]      |

### Multi-Agent Systems for Applications

AWorld's plug-and-play MAS architecture enables **real-world web application development** beyond agent training.

Build production-ready systems that handle complex tasks through:

*   **Code generation & execution**
*   **Browser automation & tool use**
*   **Multimodal understanding & generation**
*   And many more to emerge!

See [Appendix: Web Client Usage](#appendix-web-client-usage) for GAIA implementation examples.

---

## Contributing

Join us in building and improving AWorld!  Your contributions are welcome.

```bibtex
@misc{yu2025aworldorchestratingtrainingrecipe,
      title={AWorld: Orchestrating the Training Recipe for Agentic AI}, 
      author={Chengyue Yu and Siyuan Lu and Chenyi Zhuang and Dong Wang and Qintong Wu and Zongyue Li and Runsheng Gan and Chunfeng Wang and Siqi Hou and Gaochi Huang and Wenlong Yan and Lifeng Hong and Aohui Xue and Yanfeng Wang and Jinjie Gu and David Tsai and Tao Lin},
      year={2025},
      eprint={2508.20404},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.20404}, 
}
```

---

## Star History

![](https://api.star-history.com/svg?repos=inclusionAI/AWorld&type=Date)

---

## Appendix: Web Client Usage

![GAIA Agent Runtime Demo](readme_assets/gaia_demo.gif)

### Project Structure

```text
agent-project-root-dir/
    agent_deploy/
      my_first_agent/
        __init__.py
        agent.py
```

### Steps

1.  Create project folders:

    ```shell
    mkdir my-aworld-project && cd my-aworld-project # project-root-dir
    mkdir -p agent_deploy/my_first_agent
    ```

2.  Define Your Agent

    *   `__init__.py`: Create empty `__ini__.py` file.

        ```shell
        cd agent_deploy/my_first_agent
        touch __init__.py
        ```

    *   `agent.py`: Define your agent logic:

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

3.  Run Agent

    *   Setup environment variables:

        ```shell
        # Navigate back to project root
        cd ${agent-project-root-dir}

        # Set your LLM credentials
        export LLM_MODEL_NAME="gpt-4"
        export LLM_API_KEY="your-api-key-here"
        export LLM_BASE_URL="https://api.openai.com/v1"  # Optional for OpenAI
        ```

    *   Launch Your Agent:

        ```shell
        # Option 1: Launch with Web UI
        aworld web
        # Then open http://localhost:8000 in your browser

        # Option 2: Launch REST API (For integrations)
        aworld api
        # Then visit http://localhost:8000/docs for API documentation
        ```

    Success! Your agent is now running and ready to chat!