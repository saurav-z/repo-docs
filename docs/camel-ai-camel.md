<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/banner.png" alt="Banner">
  </a>
</div>

<div align="center">
    <a href="https://github.com/camel-ai/camel">
        <img src="https://img.shields.io/github/stars/camel-ai/camel?label=Stars&style=social" alt="GitHub Stars">
    </a>
    <a href="https://discord.camel-ai.org/">
        <img src="https://img.shields.io/discord/1082486657678311454?logo=discord&label=Discord&labelColor=%235462eb&logoColor=%23f5f5f5&color=%235462eb" alt="Discord">
    </a>
    <a href="https://x.com/CamelAIOrg">
        <img src="https://img.shields.io/twitter/follow/CamelAIOrg?style=social" alt="Twitter">
    </a>
    <a href="https://pypi.org/project/camel-ai">
        <img src="https://img.shields.io/pypi/dm/camel-ai?label=Downloads" alt="PyPI Downloads">
    </a>
     <a href="https://github.com/camel-ai/camel/blob/master/licenses/LICENSE">
        <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
    </a>
    <a href="https://github.com/camel-ai/camel">
        <img src="https://img.shields.io/badge/View%20on-GitHub-blue?logo=github" alt="GitHub">
    </a>
</div>

<hr>

## CAMEL: Unleash the Power of Collaborative AI Agents!

CAMEL is an open-source framework for building and exploring multi-agent systems, enabling researchers and developers to study agent behavior, scale, and potential. [Explore the original repository](https://github.com/camel-ai/camel).

**Key Features:**

*   **Large-Scale Agent Systems:** Simulate up to millions of agents to observe emergent behaviors and understand scaling laws.
*   **Dynamic Communication:** Facilitate real-time interactions between agents for complex task solving.
*   **Stateful Memory:** Equip agents with memory to enhance decision-making through historical context.
*   **Code-as-Prompt:** Leverage code and comments as prompts, making interactions intuitive for both humans and agents.
*   **Diverse Agent Types:** Support a wide range of agent roles, tasks, models, and environments.
*   **Data Generation & Tool Integration:** Automate data creation and easily integrate with multiple tools.

## Why Choose CAMEL?

CAMEL empowers researchers and developers by offering:

*   **Community Support:** Benefit from a collaborative research collective of 100+ researchers.
*   **Reproducible Research:** Utilize standardized benchmarks for reliable comparisons.
*   **Versatile Applications:** Create solutions in various domains, from data generation to world simulation.

### CAMEL Framework Design Principles

*   **Evolvability:** The framework supports continuous evolution through data generation and environment interaction.
*   **Scalability:** Designed for systems with millions of agents, ensuring efficient coordination.
*   **Statefulness:** Agents can maintain stateful memory, for complex, multi-step tasks.
*   **Code-as-Prompt:** Code and comments serve as prompts, making interaction easier for humans and agents.

## What Can You Build with CAMEL?

Explore various applications with CAMEL:

### 1. Data Generation
<div align="center">
  <a href="https://github.com/camel-ai/camel/blob/master/camel/datagen/cot_datagen.py">
    <img src="docs/images/cot.png" alt="CoT Data Generation">
  </a>
</div>
<div align="center">
  <a href="https://github.com/camel-ai/camel/tree/master/camel/datagen/self_instruct">
    <img src="docs/images/self_instruct.png" alt="Self-Instruct Data Generation">
  </a>
</div>
<div align="center">
  <a href="https://github.com/camel-ai/camel/tree/master/camel/datagen/source2synth">
    <img src="docs/images/source2synth.png" alt="Source2Synth Data Generation">
  </a>
</div>
<div align="center">
  <a href="https://github.com/camel-ai/camel/blob/master/camel/datagen/self_improving_cot.py">
    <img src="docs/images/self_improving.png" alt="Self-Improving Data Generation">
  </a>
</div>

### 2. Task Automation
<div align="center">
  <a href="https://github.com/camel-ai/camel/blob/master/camel/societies/role_playing.py">
    <img src="docs/images/role_playing.png" alt="Role Playing">
  </a>
</div>
<div align="center">
  <a href="https://github.com/camel-ai/camel/tree/master/camel/societies/workforce">
    <img src="docs/images/workforce.png" alt="Workforce">
  </a>
</div>
<div align="center">
  <a href="https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html">
    <img src="docs/images/rag_pipeline.png" alt="RAG Pipeline">
  </a>
</div>

### 3. World Simulation
<div align="center">
  <a href="https://github.com/camel-ai/oasis">
    <img src="docs/images/oasis_case.png" alt="Oasis Case">
  </a>
</div>

## Quick Start

Easily install CAMEL using pip:

```bash
pip install camel-ai
```

**Example: ChatAgent with DuckDuckGo Search**

1.  **Install the tools package:**
    ```bash
    pip install 'camel-ai[web_tools]'
    ```

2.  **Set your OpenAI API key:**
    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

3.  **Run the Python code:**
    ```python
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType
    from camel.agents import ChatAgent
    from camel.toolkits import SearchToolkit

    model = ModelFactory.create(
      model_platform=ModelPlatformType.OPENAI,
      model_type=ModelType.GPT_4O,
      model_config_dict={"temperature": 0.0},
    )

    search_tool = SearchToolkit().search_duckduckgo

    agent = ChatAgent(model=model, tools=[search_tool])

    response_1 = agent.step("What is CAMEL-AI?")
    print(response_1.msgs[0].content)

    response_2 = agent.step("What is the Github link to CAMEL framework?")
    print(response_2.msgs[0].content)
    ```

**Explore Further:**

*   **Installation Guide:** [Installation Section](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md)
*   **CAMEL Tech Stack and Cookbooks:** [docs.camel-ai.org](https://docs.camel-ai.org)
*   **Colab Demo:** [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing)

## Key Modules

Core modules to build, operate, and enhance CAMEL-AI agents and societies.

| Module | Description |
|:---|:---|
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)** | Core agent architectures and behaviors. |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)** | Manage multi-agent systems and collaboration. |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)** | Synthetic data creation and augmentation tools. |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)** | Agent intelligence model architectures. |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)** | Integrate tools for agent tasks. |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)** | Agent state management. |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)** | Persistent storage for agent data and states. |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)** | Performance evaluation frameworks. |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)** | Code and command interpretation. |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)** | Data ingestion and preprocessing. |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)** | Knowledge retrieval and RAG components. |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)** | Execution environment and process management. |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Interactive components. |

## Research

Explore our research projects:

<div align="center">
  <a href="https://crab.camel-ai.org/">
    <img src="docs/images/crab.png" alt="CRAB">
  </a>
</div>
<div align="center">
  <a href="https://agent-trust.camel-ai.org/">
    <img src="docs/images/agent_trust.png" alt="Agent Trust">
  </a>
</div>
<div align="center">
  <a href="https://oasis.camel-ai.org/">
    <img src="docs/images/oasis.png" alt="OASIS">
  </a>
</div>
<div align="center">
  <a href="https://emos-project.github.io/">
    <img src="docs/images/emos.png" alt="Emos">
  </a>
</div>

>### Research with Us
>We invite you to use CAMEL for your research. Join our community! [Reach out via email](mailto:camel-ai@eigent.ai) for more information.
>
><div align="center">
>    <img src="docs/images/partners.png" alt="Partners">
></div>

## Synthetic Datasets

### 1. Utilize Various LLMs as Backends

For more details, please see our [`Models Documentation`](https://docs.camel-ai.org/key_modules/models.html#).

> **Data (Hosted on Hugging Face)**

| Dataset        | Chat format                                                                                         | Instruction format                                                                                               | Chat format (translated)                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **AI Society** | [Chat format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_chat.tar.gz) | [Instruction format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_instructions.json) | [Chat format (translated)](https://huggingface.co/datasets/camel-ai/ai_society_translated) |
| **Code**       | [Chat format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_chat.tar.gz)             | [Instruction format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_instructions.json)             | x                                                                                          |
| **Math**       | [Chat format](https://huggingface.co/datasets/camel-ai/math)                                        | x                                                                                                                | x                                                                                          |
| **Physics**    | [Chat format](https://huggingface.co/datasets/camel-ai/physics)                                     | x                                                                                                                | x                                                                                          |
| **Chemistry**  | [Chat format](https://huggingface.co/datasets/camel-ai/chemistry)                                   | x                                                                                                                | x                                                                                          |
| **Biology**    | [Chat format](https://huggingface.co/datasets/camel-ai/biology)                                     | x                                                                                                                | x                                                                                          |

### 2. Visualizations of Instructions and Tasks

| Dataset          | Instructions                                                                                                         | Tasks                                                                                                         |
|------------------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **AI Society**   | [Instructions](https://atlas.nomic.ai/map/3a559a06-87d0-4476-a879-962656242452/db961915-b254-48e8-8e5c-917f827b74c6) | [Tasks](https://atlas.nomic.ai/map/cb96f41b-a6fd-4fe4-ac40-08e101714483/ae06156c-a572-46e9-8345-ebe18586d02b) |
| **Code**         | [Instructions](https://atlas.nomic.ai/map/902d6ccb-0bbb-4294-83a8-1c7d2dae03c8/ace2e146-e49f-41db-a1f4-25a2c4be2457) | [Tasks](https://atlas.nomic.ai/map/efc38617-9180-490a-8630-43a05b35d22d/2576addf-a133-45d5-89a9-6b067b6652dd) |
| **Misalignment** | [Instructions](https://atlas.nomic.ai/map/5c491035-a26e-4a05-9593-82ffb2c3ab40/2bd98896-894e-4807-9ed8-a203ccb14d5e) | [Tasks](https://atlas.nomic.ai/map/abc357dd-9c04-4913-9541-63e259d7ac1f/825139a4-af66-427c-9d0e-f36b5492ab3f) |

## Cookbooks (Usecases)

Practical guides and tutorials for implementing CAMEL-AI functionalities.

### 1. Basic Concepts
| Cookbook | Description |
|:---|:---|
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)** | A step-by-step guide. |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)** | Build a collaborative society. |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)** | Best practices for message handling. |

### 2. Advanced Features
| Cookbook | Description |
|:---|:---|
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)** | Integrate tools for enhanced functionality. |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)** | Implement memory systems. |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)** | Retrieval-Augmented Generation recipes. |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)** | Leverage knowledge graphs with RAG. |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)** | Tools for tracking and managing agents. |

### 3. Model Training & Data Generation
| Cookbook | Description |
|:---|:---|
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)** | Generate data and fine-tune with Unsloth. |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)** | Generate data with function calls. |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)** | Generate and upload CoT data. |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)** | Generate CoT data and SFT Qwen. |

### 4. Multi-Agent Systems & Applications
| Cookbook | Description |
|:---|:---|
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)** | Role-playing agents for data scraping. |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)** | Build a team of agents. |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** | Build dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system. |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)** | Build a customer service bot for Discord using Agentic RAG. |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)** | Build a customer service bot for Discord that supports local deployment. |

### 5. Data Processing
| Cookbook | Description |
|:---|:---|
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)** | Agents in video data analysis. |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Ingest data from websites. |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Create AI agents with PDFs. |

## Real-World Usecases

Explore real-world applications demonstrating the value of CAMEL's multi-agent framework.

### 1 Infrastructure Automation

| Usecase | Description |
| --- | --- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)** | Real-world usecases demonstrating how CAMEL’s multi-agent framework enables real business value across infrastructure automation, productivity workflows, retrieval-augmented conversations, intelligent document/video analysis, and collaborative research. |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** | Intelligent agents manage Cloudflare resources dynamically. |

### 2 Productivity & Business Workflows

| Usecase | Description |
| --- | --- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)** | Coordinate agents to manage Airbnb listings. |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Analyze PowerPoint documents. |

### 3 Retrieval-Augmented Multi-Agent Chat

| Usecase | Description |
| --- | --- |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)** | Query and understand GitHub codebases. |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)** | Conversational agents summarize video transcripts. |

### 4 Video & Document Intelligence

| Usecase | Description |
| --- | --- |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | Agents perform OCR on video screenshots. |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | CAMEL agents use OCR with Mistral to analyze documents. |

### 5 Research & Collaboration

| Usecase | Description |
| --- | --- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Research agents collaborating on literature review. |

## 🗓️ Events

Stay involved with the CAMEL community!

-   🎙️ **Community Meetings**
-   🏆 **Competitions**
-   🤝 **Volunteer Activities**
-   🌍 **Ambassador Programs**

> Want to host or participate in a CAMEL event? Join our [Discord](https://discord.com/invite/CNcNpquyDc) or apply for our [Ambassador Program](https://www.camel-ai.org/ambassador).

## Contributing to CAMEL

We welcome contributions! Review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md).

## Community & Contact

*   **GitHub Issues:** [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** [Learn more](https://www.camel-ai.org/community)
*   **WeChat Community:** Scan the QR code below.

<div align="center">
    <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
</div>

## Citation

```
@inproceedings{li2023camel,
  title={CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society},
  author={Li, Guohao and Hammoud, Hasan Abed Al Kader and Itani, Hani and Khizbullin, Dmitrii and Ghanem, Bernard},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Acknowledgment

Thanks to [Nomic AI](https://home.nomic.ai/), Haya Hammoud, and the creators of the research ideas used in our modules, which include :
- `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/). [[Example](https://github.com/camel-ai/camel/blob/master/examples/ai_society/babyagi_playing.py)]

- `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094). [[Example](https://github.com/camel-ai/camel/blob/master/examples/personas/personas_generation.py)]

- `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560). [[Example](https://github.com/camel-ai/camel/blob/master/examples/datagen/self_instruct/self_instruct.py)]
## License

Licensed under Apache 2.0.