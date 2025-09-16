<div align="center">
  <a href="https://github.com/camel-ai/camel">
    <img src="docs/images/banner.png" alt="CAMEL Banner">
  </a>
</div>

</br>

<div align="center">

[![Documentation][docs-image]][docs-url]
[![Discord][discord-image]][discord-url]
[![X (Twitter)][x-image]][x-url]
[![Reddit][reddit-image]][reddit-url]
[![Wechat][wechat-image]][wechat-url]
[![Hugging Face][huggingface-image]][huggingface-url]
[![GitHub Stars][star-image]][star-url]
[![Package License][package-license-image]][package-license-url]
[![PyPI Downloads][package-download-image]][package-download-url]

<a href="https://trendshift.io/repositories/649" target="_blank"><img src="https://trendshift.io/api/badge/repositories/649" alt="camel-ai/camel | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

<hr>

<div align="center">
  <h4 align="center">
    <a href="https://github.com/camel-ai/camel#community">Community</a> |
    <a href="https://github.com/camel-ai/camel#installation">Installation</a> |
    <a href="https://github.com/camel-ai/camel/tree/HEAD/examples">Examples</a> |
    <a href="https://arxiv.org/abs/2303.17760">Paper</a> |
    <a href="https://github.com/camel-ai/camel#citation">Citation</a> |
    <a href="https://github.com/camel-ai/camel#contributing-to-camel-">Contributing</a> |
    <a href="https://www.camel-ai.org/">CAMEL-AI</a>
  </h4>
</div>

<p style="line-height: 1.5; text-align: center;">
  <b>CAMEL, an open-source framework, unlocks the power of multi-agent systems for cutting-edge research and real-world applications.</b> CAMEL empowers researchers and developers to explore the scaling laws of agents by implementing and supporting diverse agents, tasks, prompts, models, and simulated environments.
</p>

<div align="center">
  Join the CAMEL community on <a href="https://discord.camel-ai.org/">Discord</a> or <a href="https://ghli.org/camel/wechat.png">WeChat</a> to push the boundaries of agent technology.
  <br>
  🌟 Star CAMEL on GitHub to stay updated on new releases and developments!
</div>

<div align="center">
    <img src="docs/images/stars.gif" alt="Star">
  </a>
</div>

<br>

<details>
<summary><kbd>Table of Contents</kbd></summary>

- [Key Features](#key-features)
- [Why Use CAMEL?](#why-use-camel)
- [What Can You Build with CAMEL?](#what-can-you-build-with-camel)
    - [1. Data Generation](#1-data-generation)
    - [2. Task Automation](#2-task-automation)
    - [3. World Simulation](#3-world-simulation)
- [Quick Start](#quick-start)
    - [Starting with ChatAgent](#starting-with-chatagent)
    - [Seeking Help](#seeking-help)
- [Tech Stack](#tech-stack)
    - [Key Modules](#key-modules)
- [Research](#research)
- [Synthetic Datasets](#synthetic-datasets)
- [Cookbooks (Usecases)](#cookbooks-usecases)
    - [1. Basic Concepts](#1-basic-concepts)
    - [2. Advanced Features](#2-advanced-features)
    - [3. Model Training & Data Generation](#3-model-training--data-generation)
    - [4. Multi-Agent Systems & Applications](#4-multi-agent-systems--applications)
    - [5. Data Processing](#5-data-processing)
- [Real-World Usecases](#real-world-usecases)
- [🧱 Built with CAMEL (Real-world Products & Research)](#-built-with-camel-real-world-products--research)
    - [Research Projects](#research-projects)
    - [Product Projects](#product-projects)
- [🗓️ Events](#️-events)
- [Contributing to CAMEL](#contributing-to-camel)
- [Community & Contact](#community--contact)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)
- [License](#license)

</details>

## Key Features

*   **Large-Scale Agent Systems:** Simulate up to millions of agents for emergent behavior studies.
*   **Dynamic Communication:** Facilitate real-time agent interactions for complex task collaboration.
*   **Stateful Memory:** Enable agents to retain and utilize historical context for improved decision-making.
*   **Multiple Benchmarks:** Utilize standardized benchmarks for rigorous performance evaluation and comparisons.
*   **Diverse Agent Types:** Support various agent roles, tasks, models, and environments.
*   **Data Generation & Tool Integration:** Automate data creation and seamlessly integrate with multiple tools.

## Why Use CAMEL?

CAMEL is a community-driven research framework, with over 100 researchers, for advancing research in Multi-Agent Systems, offering these key advantages:

*   **Scalability:** Designed for systems with millions of agents.
*   **Evolvability:** Supports continuous system evolution through data generation and environmental interaction.
*   **Code-as-Prompt:** Code and comments serve as prompts, fostering interpretable code.
*   **Statefulness:** Enables multi-step interactions and complex task handling.

## What Can You Build with CAMEL?

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

<br>

## Quick Start

Get started with CAMEL in minutes:

```bash
pip install camel-ai
```

### Starting with ChatAgent

Here's how to create a `ChatAgent` and use it with DuckDuckGo search:

1.  **Install the tools package:**

    ```bash
    pip install 'camel-ai[web_tools]'
    ```

2.  **Set up your OpenAI API key:**

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
    # CAMEL-AI is the first LLM (Large Language Model) multi-agent framework
    # and an open-source community focused on finding the scaling laws of agents.
    # ...

    response_2 = agent.step("What is the Github link to CAMEL framework?")
    print(response_2.msgs[0].content)
    # The GitHub link to the CAMEL framework is
    # [https://github.com/camel-ai/camel](https://github.com/camel-ai/camel).
    ```

For detailed instructions, see the [installation guide](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).

Explore the [CAMEL Tech Stack and Cookbooks](https://docs.camel-ai.org) to build advanced multi-agent systems.

Explore these resources:

*   **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**
*   **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)**
*   **[Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)**
*   **[Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)**

### Seeking Help

Get assistance on [CAMEL discord](https://discord.camel-ai.org/).

<br>

## Tech Stack

<div align="center">
  <a href="https://docs.camel-ai.org">
    <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="TechStack">
  </a>
</div>

### Key Modules

Core components for building and enhancing CAMEL-AI agents and societies.

| Module | Description |
| :------------------------------------------------------------------ | :---------------------------------------------------------------------------------------- |
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)** | Core agent architectures and behaviors. |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)** | Components for managing multi-agent systems. |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)** | Tools for synthetic data creation and augmentation. |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)** | Model architectures and customization options. |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)** | Integration for specialized agent tasks. |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)** | State management for agent operations. |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)** | Persistent storage solutions. |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)** | Performance evaluation and testing frameworks. |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)** | Code and command interpretation capabilities. |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)** | Data ingestion and preprocessing tools. |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)** | Knowledge retrieval and RAG components. |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)** | Execution environment and process management. |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Interactive components for human oversight. |
---

## Research

We are a community-driven research collective with 100+ researchers, exploring frontier research in Multi-agent Systems.

**Explore our research projects:**

<div align="center">
  <a href="https://github.com/camel-ai/owl">
    <img src="docs/images/owl.png" alt="OWL">
  </a>
</div>

<div align="center">
  <a href="https://oasis.camel-ai.org/">
    <img src="docs/images/oasis.png" alt="OASIS">
  </a>
</div>

<div align="center">
  <a href="https://crab.camel-ai.org/">
    <img src="docs/images/crab.png" alt="CRAB">
  </a>
</div>

<div align="center">
  <a href="https://github.com/camel-ai/loong">
    <img src="docs/images/loong.png" alt="Loong">
  </a>
</div>

<div align="center">
  <a href="https://agent-trust.camel-ai.org/">
    <img src="docs/images/agent_trust.png" alt="Agent Trust">
  </a>
</div>

<div align="center">
  <a href="https://emos-project.github.io/">
    <img src="docs/images/emos.png" alt="Emos">
  </a>
</div>

>### Research with Us
>
> We invite you to use CAMEL for impactful research.  Join our ongoing projects or test new ideas.  [Reach out via email](mailto:camel-ai@eigent.ai) for more information.
>
><div align="center">
>    <img src="docs/images/partners.png" alt="Partners">
></div>

<br>

## Synthetic Datasets

### 1. Utilize Various LLMs as Backends

For more details, please see our [`Models Documentation`](https://docs.camel-ai.org/key_modules/models.html#).

> **Data (Hosted on Hugging Face)**

| Dataset        | Chat format                                                                                         | Instruction format                                                                                               | Chat format (translated)                                                                   |
| :------------- | :-------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------- |
| **AI Society** | [Chat format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_chat.tar.gz) | [Instruction format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_instructions.json) | [Chat format (translated)](https://huggingface.co/datasets/camel-ai/ai_society_translated) |
| **Code**       | [Chat format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_chat.tar.gz)             | [Instruction format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_instructions.json)             | x                                                                                          |
| **Math**       | [Chat format](https://huggingface.co/datasets/camel-ai/math)                                        | x                                                                                                                | x                                                                                          |
| **Physics**    | [Chat format](https://huggingface.co/datasets/camel-ai/physics)                                     | x                                                                                                                | x                                                                                          |
| **Chemistry**  | [Chat format](https://huggingface.co/datasets/camel-ai/chemistry)                                   | x                                                                                                                | x                                                                                          |
| **Biology**    | [Chat format](https://huggingface.co/datasets/camel-ai/biology)                                     | x                                                                                                                | x                                                                                          |

### 2. Visualizations of Instructions and Tasks

| Dataset          | Instructions                                                                                                         | Tasks                                                                                                         |
| :--------------- | :--------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **AI Society**   | [Instructions](https://atlas.nomic.ai/map/3a559a06-87d0-4476-a879-962656242452/db961915-b254-48e8-8e5c-917f827b74c6) | [Tasks](https://atlas.nomic.ai/map/cb96f41b-a6fd-4fe4-ac40-08e101714483/ae06156c-a572-46e9-8345-ebe18586d02b) |
| **Code**         | [Instructions](https://atlas.nomic.ai/map/902d6ccb-0bbb-4294-83a8-1c7d2dae03c8/ace2e146-e49f-41db-a1f4-25a2c4be2457) | [Tasks](https://atlas.nomic.ai/map/efc38617-9180-490a-8630-43a05b35d22d/2576addf-a133-45d5-89a9-6b067b6652dd) |
| **Misalignment** | [Instructions](https://atlas.nomic.ai/map/5c491035-a26e-4a05-9593-82ffb2c3ab40/2bd98896-894e-4807-9ed8-a203ccb14d5e) | [Tasks](https://atlas.nomic.ai/map/abc357dd-9c04-4913-9541-63e259d7ac1f/825139a4-af66-427c-9d0e-f36b5492ab3f) |

<br>

## Cookbooks (Usecases)

Practical guides and tutorials for implementing specific functionalities.

### 1. Basic Concepts

| Cookbook | Description |
| :---------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)** | Build your first agent. |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)** | Build a collaborative agent society. |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)** | Best practices for agent message handling. |

### 2. Advanced Features

| Cookbook | Description |
| :---------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)** | Integrate tools for enhanced functionality. |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)** | Implement memory systems in agents. |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)** | Recipes for Retrieval-Augmented Generation. |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)** | Leverage knowledge graphs with RAG. |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)** | Tools for agent tracking and management. |

### 3. Model Training & Data Generation

| Cookbook | Description |
| :--------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- |
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)** | Generate data and fine-tune models with Unsloth. |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)** | Generate data with function calls and the Hermes format. |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)** | Generate CoT data and upload to Huggingface. |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)** | Generate CoT data, SFT Qwen, and upload to Huggingface. |

### 4. Multi-Agent Systems & Applications

| Cookbook | Description |
| :---------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)** | Create role-playing agents for scraping and reporting. |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)** | Build a team of agents for collaborative judging. |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** | Builds dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system. |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)** | Build a customer service bot for Discord with Agentic RAG. |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)** | Build a local deployment customer service bot for Discord using Agentic RAG. |

### 5. Data Processing

| Cookbook | Description |
| :------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------- |
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)** | Agent techniques for video data analysis. |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Ingesting data from websites using Firecrawl. |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Create AI agents that work with your PDFs using Chunkr and Mistral AI. |

<br>

## Real-World Usecases

See how CAMEL’s multi-agent framework delivers real business value:

### 1 Infrastructure Automation

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)** | Real-world usecases demonstrating how CAMEL’s multi-agent framework enables real business value across infrastructure automation, productivity workflows, retrieval-augmented conversations, intelligent document/video analysis, and collaborative research. |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** | Intelligent agents manage Cloudflare resources dynamically. |

### 2 Productivity & Business Workflows

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)** | Coordinate agents to optimize Airbnb listings. |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Analyze and extract insights from PowerPoint documents. |

### 3 Retrieval-Augmented Multi-Agent Chat

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)** | Query and understand GitHub codebases with RAG. |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)** | Extract and summarize video transcripts. |

### 4 Video & Document Intelligence

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | OCR for video content summarization. |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | Analyze documents with OCR using Mistral. |

### 5 Research & Collaboration

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Simulate a team of research agents for literature review. |

<br>

## 🧱 Built with CAMEL (Real-world Products & Research)
<div align="left">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/built_with_CAMEL.png" alt="Built with CAMEL" height="40px">
  </a>
</div>

### Research Projects

| Name | Description |
| :------------------------------------------------------------------------ | :------------------------------------------------------- |
| **[ChatDev](https://github.com/OpenBMB/ChatDev/tree/main/camel)** | Communicative Agents for software Development |
| **[Paper2Poster](https://github.com/Paper2Poster/Paper2Poster)** | Multimodal poster automation from scientific papers |

### Product Projects

| Name | Description |
| :-------------------------------------------------------------- | :------------------------------------------------ |
| **[Eigent](https://www.eigent.ai/)** | The World First Multi-agent Workforce |
| **[EigentBot](https://bot.eigent.ai/)** | One EigentBot, Every Code Answer |
| **[Matrix](https://matrix.eigent.ai/)** | Social Media Simulation |
| **[AI Geometric](https://www.linkedin.com/posts/aigeometric_ai-interviewpreparation-careerdevelopment-activity-7261428422516555776-MtaK/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAChHluEB9xRwkjiJ6VSAzqM2Y-U4NI2sKGY)** | AI-powered interview copilot |
| **[Log10](https://github.com/log10-io/log10/blob/main/src/log10/agents/camel.py)** | AI accuracy, delivered |

## 🗓️ Events

CAMEL hosts and participates in community events:

-   🎙️ **Community Meetings** — Weekly virtual syncs
-   🏆 **Competitions** — Hackathons and coding challenges
-   🤝 **Volunteer Activities** — Contributions and mentorship
-   🌍 **Ambassador Programs** — Represent CAMEL in your community

> Participate in or host a CAMEL event - join our [Discord](https://discord.com/invite/CNcNpquyDc) or the [Ambassador Program](https://www.camel-ai.org/ambassador).

## Contributing to CAMEL

> Contribute code to our open-source initiative - review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md).

> Support CAMEL by sharing it on social media, at events, or during conferences!

<br>

## Community & Contact

For more information please contact camel-ai@eigent.ai.

*   **GitHub Issues:** Report bugs, request features: [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** Get real-time support and connect with the community: [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** Stay updated: [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** Advocate for CAMEL-AI: [Learn more](https://www.camel-ai.org/community)
*   **WeChat Community:** Scan the QR code below to join our WeChat community.

  <div align="center">
    <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
  </div>

<br>

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

Special thanks to [Nomic AI](https://home.nomic.ai/) for Atlas access.

We also thank Haya Hammoud for designing the initial project logo.

If you use these modules, please cite the original works:

*   `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/). [[Example](https://github.com/camel-ai/camel/blob/master/examples/ai_society/babyagi_playing.py)]
*   `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094). [[Example](https://github.com/camel-ai/camel/blob/master/examples/personas/personas_generation.py)]
*   `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560). [[Example](https://github.com/camel-ai/camel/blob/master/examples/datagen/self_instruct/self_instruct.py)]

## License

The source code is licensed under Apache 2.0.

<br>

[docs-image]: https://img.shields.io/badge/Documentation-EB3ECC
[docs-url]: https://camel-ai.github.io/camel/index.html
[star-image]: https://img.shields.io/github/stars/camel-ai/camel?label=stars&logo=github&color=brightgreen
[star-url]: https://github.com/camel-ai/camel/stargazers
[package-license-image]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[package-license-url]: https://github.com/camel-ai/camel/blob/master/licenses/LICENSE
[package-download-image]: https://img.shields.io/pypi/dm/camel-ai

[colab-url]: https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing
[colab-image]: https://colab.research.google.com/assets/colab-badge.svg
[huggingface-url]: https://huggingface.co/camel-ai
[huggingface-image]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CAMEL--AI-ffc107?color=ffc107&logoColor=white
[discord-url]: https://discord.camel-ai.org/
[discord-image]: https://img.shields.io/discord/1082486657678311454?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f