<div align="center">
  <a href="https://github.com/camel-ai/camel">
    <img src="docs/images/banner.png" alt="CAMEL Banner">
  </a>
</div>

<br>

# CAMEL: The Multi-Agent Framework Revolutionizing AI Agent Research

CAMEL empowers researchers and developers to explore the scaling laws of AI agents, fostering innovation in multi-agent systems.  Explore the [CAMEL GitHub repository](https://github.com/camel-ai/camel) to get started today!

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
</div>

<div align="center">
  <a href="https://trendshift.io/repositories/649" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/649" alt="camel-ai/camel | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

<br>

<div align="center">
  [Community](https://github.com/camel-ai/camel#community) |
  [Installation](https://github.com/camel-ai/camel#installation) |
  [Examples](https://github.com/camel-ai/camel/tree/HEAD/examples) |
  [Paper](https://arxiv.org/abs/2303.17760) |
  [Citation](https://github.com/camel-ai/camel#citation) |
  [Contributing](https://github.com/camel-ai/camel#contributing-to-camel) |
  [CAMEL-AI Website](https://www.camel-ai.org/)
</div>

<br>

<div align="center">
  <p style="line-height: 1.5; text-align: center;">
    CAMEL is an open-source framework and vibrant community dedicated to understanding and advancing the capabilities of AI agents through large-scale experiments. We provide the tools and support needed to explore agent behaviors, potential risks, and scaling laws by supporting various agent types, tasks, prompts, models, and simulated environments.
  </p>
  <p>
    Join our community and help us push the boundaries of AI agents!  Connect with us on <a href="https://discord.camel-ai.org/">Discord</a> or <a href="https://ghli.org/camel/wechat.png">WeChat</a>.
  </p>
  <p>
    🌟 Star CAMEL on GitHub to stay up-to-date on new releases and developments.
  </p>
</div>

<div align="center">
    <img src="docs/images/stars.gif" alt="Star">
</div>

<br>

<details>
  <summary><kbd>Table of Contents</kbd></summary>
  <br/>
  - [CAMEL Framework Design Principles](#camel-framework-design-principles)
  - [Why Use CAMEL for Your Research?](#why-use-camel-for-your-research)
  - [Key Features](#key-features)
  - [What Can You Build With CAMEL?](#what-can-you-build-with-camel)
    - [Data Generation](#1-data-generation)
    - [Task Automation](#2-task-automation)
    - [World Simulation](#3-world-simulation)
  - [Quick Start](#quick-start)
    - [Starting with ChatAgent](#starting-with-chatagent)
    - [Seeking Help](#seeking-help)
  - [Tech Stack](#tech-stack)
    - [Key Modules](#key-modules)
  - [Research](#research)
  - [Synthetic Datasets](#synthetic-datasets)
  - [Cookbooks (Usecases)](#cookbooks-usecases)
    - [Basic Concepts](#1-basic-concepts)
    - [Advanced Features](#2-advanced-features)
    - [Model Training & Data Generation](#3-model-training--data-generation)
    - [Multi-Agent Systems & Applications](#4-multi-agent-systems--applications)
    - [Data Processing](#5-data-processing)
  - [Real-World Usecases](#real-world-usecases)
  - [🧱 Built with CAMEL (Real-world Products & Research)](#-built-with-camel-real-world-producs--research)
    - [Research Projects](#research-projects)
    - [Product Projects](#product-projects)
  - [🗓️ Events](#️-events)
  - [Contributing to CAMEL](#contributing-to-camel)
  - [Community & Contact](#community--contact)
  - [Citation](#citation)
  - [Acknowledgment](#acknowledgment)
  - [License](#license)
  <br/>
</details>

<br>

## CAMEL Framework Design Principles

### 🧬 Evolvability

The framework enables continuous evolution of multi-agent systems through data generation and environment interaction.  This can be driven by reinforcement learning with verifiable rewards or supervised learning.

### 📈 Scalability

Designed to support systems with millions of agents, ensuring efficient coordination, communication, and resource management at scale.

### 💾 Statefulness

Agents maintain stateful memory, enabling them to perform multi-step interactions with environments and tackle sophisticated tasks effectively.

### 📖 Code-as-Prompt

Code and comments act as prompts for agents.  Clear and readable code is crucial for both human and agent understanding.

<br>

## Why Use CAMEL for Your Research?

CAMEL is a leading open-source framework, backed by a community of 100+ researchers, accelerating research in Multi-Agent Systems.  Researchers choose CAMEL for:

## Key Features

*   **Large-Scale Agent Systems:** Simulate up to 1 million agents.
*   **Dynamic Communication:** Enable real-time agent interaction.
*   **Stateful Memory:** Empower agents to retain and use historical context.
*   **Multiple Benchmark Support:** Rigorous evaluation and comparison.
*   **Diverse Agent Types:** Support for various roles, models, and environments.
*   **Data Generation & Tool Integration:** Streamline research workflows.

<br>

## What Can You Build With CAMEL?

CAMEL empowers you to build a wide range of AI applications:

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

Get started with CAMEL quickly using pip:

```bash
pip install camel-ai
```

### Starting with ChatAgent

This example demonstrates creating a `ChatAgent` with the CAMEL framework using DuckDuckGo for web searches.

1.  **Install web tools:**

    ```bash
    pip install 'camel-ai[web_tools]'
    ```

2.  **Set your OpenAI API key:**

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

    or, use a `.env` file:

    ```bash
    cp .env.example .env
    # then edit .env and add your keys
    ```

3.  **Run this Python code:**

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

For complete instructions, see the [installation section](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).

Explore our [CAMEL Tech Stack and Cookbooks](https://docs.camel-ai.org) to build advanced multi-agent systems!

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)][colab-url]

Experiment with different agent types:

*   [Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)
*   [Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)
*   [Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)
*   [Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)

### Seeking Help

Join the [CAMEL Discord](https://discord.camel-ai.org/) for support!

<br>

## Tech Stack

<div align="center">
  <a href="https://docs.camel-ai.org">
    <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="TechStack">
  </a>
</div>

### Key Modules

Core components for building, operating, and enhancing CAMEL-AI agents and societies.

| Module | Description |
|:---|:---|
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)** | Core agent architectures and behaviors. |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)** | Building and managing multi-agent systems. |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)** | Tools for synthetic data creation. |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)** | Model architectures and customization. |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)** | Tools integration for specialized agent tasks. |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)** | Agent state management. |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)** | Persistent storage for agent data. |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)** | Performance evaluation and testing. |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)** | Code and command interpretation. |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)** | Data ingestion and preprocessing. |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)** | Knowledge retrieval and RAG components. |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)** | Execution environment management. |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Interactive components for human oversight. |
---

## Research

CAMEL fosters research into agent behavior, capabilities, and risks at scale.

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
>We invite you to use CAMEL for impactful research!
>
>We are a community-driven research collective with 100+ researchers exploring the frontier research of Multi-agent Systems. Join our ongoing projects or test new ideas with us, [reach out via email](mailto:camel-ai@eigent.ai) for more information.
>
><div align="center">
>    <img src="docs/images/partners.png" alt="Partners">
></div>

<br>

## Synthetic Datasets

### 1. Utilize Various LLMs as Backends

See our [`Models Documentation`](https://docs.camel-ai.org/key_modules/models.html#) for details.

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

<br>

## Cookbooks (Usecases)

Practical guides and tutorials for implementing functionalities in CAMEL-AI agents and societies.

### 1. Basic Concepts

| Cookbook | Description |
|:---|:---|
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)** | Step-by-step guide to building your first agent. |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)** | Learn to build a collaborative society of agents. |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)** | Best practices for message handling. |

### 2. Advanced Features

| Cookbook | Description |
|:---|:---|
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)** | Integrating tools for enhanced functionality. |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)** | Implementing memory systems. |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)** | Recipes for Retrieval-Augmented Generation. |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)** | Leveraging knowledge graphs with RAG. |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)** | Tools for tracking and managing agents. |

### 3. Model Training & Data Generation

| Cookbook | Description |
|:---|:---|
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)** | Generate data with CAMEL and fine-tune models with Unsloth. |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)** | Generate data with real function calls and the Hermes format. |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)** | Generate CoT data with CAMEL and upload to Huggingface. |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)** | Generate CoT data, SFT Qwen with Unsolth, and upload to Huggingface. |

### 4. Multi-Agent Systems & Applications

| Cookbook | Description |
|:---|:---|
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)** | Create role-playing agents for data scraping. |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)** | Build a team of agents for judging. |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** | Build dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system. |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)** | Build a customer service bot for Discord using Agentic RAG. |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)** | Build a local customer service Discord bot using Agentic RAG. |

### 5. Data Processing

| Cookbook | Description |
|:---|:---|
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)** | Techniques for agents in video data analysis. |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Extracting and processing data from websites. |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Create AI agents to work with PDFs. |

<br>

## Real-World Usecases

CAMEL's multi-agent framework unlocks real-world business value.

### 1 Infrastructure Automation

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)** | Intelligent agents automate and manage ACI resources. |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** | Agents manage Cloudflare resources, enabling scalable cloud security and performance tuning. |

### 2 Productivity & Business Workflows

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)** | Optimize and manage Airbnb listings and host operations. |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Extract insights from PowerPoint documents through multi-agent collaboration. |

### 3 Retrieval-Augmented Multi-Agent Chat

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)** | Query and understand GitHub codebases through CAMEL agents leveraging RAG-style workflows. |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)** | Agents extract and summarize video transcripts. |

### 4 Video & Document Intelligence

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | Agents perform OCR to summarize video content. |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | CAMEL agents use OCR with Mistral to analyze documents. |

### 5 Research & Collaboration

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Research agents collaborate on literature review. |

<br>

## 🧱 Built with CAMEL (Real-world Products & Research)
<div align="left">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/built_with_CAMEL.png" alt="Built with CAMEL" height="40px">
  </a>
</div>

### Research Projects

| Name | Description |
|:---|:---|
| **[ChatDev](https://github.com/OpenBMB/ChatDev/tree/main/camel)** | Communicative Agents for software Development |
| **[Paper2Poster](https://github.com/Paper2Poster/Paper2Poster)** | Multimodal poster automation from scientific papers |

### Product Projects

| Name | Description |
|:---|:---|
| **[Eigent](https://www.eigent.ai/)** | The World First Multi-agent Workforce |
| **[EigentBot](https://bot.eigent.ai/)** | One EigentBot, Every Code Answer |
| **[Matrix](https://matrix.eigent.ai/)** | Social Media Simulation |
| **[AI Geometric](https://www.linkedin.com/posts/aigeometric_ai-interviewpreparation-careerdevelopment-activity-7261428422516555776-MtaK/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAChHluEB9xRwkjiJ6VSAzqM2Y-U4NI2sKGY)** | AI-powered interview copilot |
| **[Log10](https://github.com/log10-io/log10/blob/main/src/log10/agents/camel.py)** | AI accuracy, delivered |

<br>

## 🗓️ Events

We host and participate in community events, including:

-   🎙️ **Community Meetings:** Weekly virtual syncs.
-   🏆 **Competitions:** Hackathons, and coding challenges.
-   🤝 **Volunteer Activities:** Contributions, documentation, and mentorship.
-   🌍 **Ambassador Programs:** Represent CAMEL.

> Want to host or participate in a CAMEL event? Join our [Discord](https://discord.com/invite/CNcNpquyDc) or want to be part of [Ambassador Program](https://www.camel-ai.org/community).

<br>

## Contributing to CAMEL

> If you'd like to contribute, review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md).
>
> Help us by sharing CAMEL on social media or at events.

<br>

## Community & Contact

*   **GitHub Issues:** [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** [Learn more](https://www.camel-ai.org/community)
*   **WeChat Community:** Scan the QR code below to join.

  <div align="center">
    <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
  </div>
<br>
For more information please contact camel-ai@eigent.ai

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

Special thanks to [Nomic AI](https://home.nomic.ai/) and Haya Hammoud.

Please cite the original works if you use the modules from:
- `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/). [[Example](https://github.com/camel-ai/camel/blob/master/examples/ai_society/babyagi_playing.py)]

- `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094). [[Example](https://github.com/camel-ai/camel/blob/master/examples/personas/personas_generation.py)]

- `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560). [[Example](https://github.com/camel-ai/camel/blob/master/examples/datagen/self_instruct/self_instruct.py)]

## License

Licensed under Apache 2.0.

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
[hugging