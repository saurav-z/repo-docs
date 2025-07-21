<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/banner.png" alt="Banner">
  </a>
</div>

</br>

<div align="center">

[![Documentation][docs-image]][docs-url]
[![Discord][discord-image]][discord-url]
[![X][x-image]][x-url]
[![Reddit][reddit-image]][reddit-url]
[![Wechat][wechat-image]][wechat-url]
[![Hugging Face][huggingface-image]][huggingface-url]
[![Star][star-image]][star-url]
[![Package License][package-license-image]][package-license-url]
[![PyPI Download][package-download-image]][package-download-url]

<a href="https://trendshift.io/repositories/649" target="_blank"><img src="https://trendshift.io/api/badge/repositories/649" alt="camel-ai/camel | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

<hr>

## CAMEL: Unleash the Power of Multi-Agent Systems for Cutting-Edge AI Research

[CAMEL](https://github.com/camel-ai/camel) is a leading open-source framework empowering researchers to explore the frontier of multi-agent systems and discover the scaling laws of AI agents.

**Key Features:**

*   ✅ **Large-Scale Agent Simulation:** Simulate up to 1 million agents to observe emergent behaviors and scaling laws.
*   ✅ **Dynamic Communication:** Facilitate real-time agent interactions for seamless collaboration.
*   ✅ **Stateful Memory:** Equip agents with memory for enhanced decision-making over extended interactions.
*   ✅ **Multiple Benchmarks:** Evaluate agent performance using standardized benchmarks for reliable comparisons.
*   ✅ **Diverse Agent Types:** Work with various agent roles, tasks, models, and environments for interdisciplinary experiments.
*   ✅ **Data Generation and Tool Integration:** Automate data creation and integrate with multiple tools for streamlined research.

<br>

### Table of Contents

*   [Why Use CAMEL for Your Research?](#why-use-camel-for-your-research)
*   [What Can You Build With CAMEL?](#what-can-you-build-with-camel)
    *   [1. Data Generation](#1-data-generation)
    *   [2. Task Automation](#2-task-automation)
    *   [3. World Simulation](#3-world-simulation)
*   [Quick Start](#quick-start)
    *   [Starting with ChatAgent](#starting-with-chatagent)
*   [Tech Stack](#tech-stack)
*   [Research](#research)
*   [Synthetic Datasets](#synthetic-datasets)
*   [Cookbooks (Usecases)](#cookbooks-usecases)
    *   [1. Basic Concepts](#1-basic-concepts)
    *   [2. Advanced Features](#2-advanced-features)
    *   [3. Model Training & Data Generation](#3-model-training--data-generation)
    *   [4. Multi-Agent Systems & Applications](#4-multi-agent-systems--applications)
    *   [5. Data Processing](#5-data-processing)
*   [Real-World Usecases](#real-world-usecases)
    *   [1 Infrastructure Automation](#1-infrastructure-automation)
    *   [2 Productivity & Business Workflows](#2-productivity--business-workflows)
    *   [3 Retrieval-Augmented Multi-Agent Chat](#3-retrieval-augmented-multi-agent-chat)
    *   [4 Video & Document Intelligence](#4-video--document-intelligence)
    *   [5 Research & Collaboration](#5-research--collaboration)
*   [🗓️ Events](#️-events)
*   [Contributing to CAMEL](#contributing-to-camel)
*   [Community & Contact](#community--contact)
*   [Citation](#citation)
*   [Acknowledgment](#acknowledgment)
*   [License](#license)

<br>

## Why Use CAMEL for Your Research?

CAMEL is a community-driven research platform with over 100 researchers dedicated to pushing the boundaries of Multi-Agent Systems. Researchers choose CAMEL for its comprehensive features and collaborative environment.

<br>

## What Can You Build With CAMEL?

CAMEL provides the tools and infrastructure to build a variety of multi-agent system applications.

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

Get started with CAMEL quickly using these simple steps.

### Starting with ChatAgent

This example shows you how to create a `ChatAgent` using the CAMEL framework and query DuckDuckGo.

1.  **Install the tools package:**

    ```bash
    pip install 'camel-ai[web_tools]'
    ```

2.  **Set up your OpenAI API key:**

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

3.  **Run the following Python code:**

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

For complete installation instructions and more options, see the [installation section](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).  Then explore our CAMEL Tech Stack and Cookbooks at [docs.camel-ai.org](https://docs.camel-ai.org) to build powerful multi-agent systems.

<details>
  <summary>
    Demo
  </summary>

  We provide a [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing) demo showcasing a conversation between two ChatGPT agents playing roles as a python programmer and a stock trader collaborating on developing a trading bot for stock market.

</details>

Explore different types of agents, their roles, and their applications.

-   [Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)
-   [Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)
-   [Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)
-   [Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)

### Seeking Help

Please reach out to us on [CAMEL discord](https://discord.camel-ai.org/) if you encounter any issues setting up CAMEL.

<br>

## Tech Stack

<div align="center">
  <a href="https://docs.camel-ai.org">
    <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="TechStack">
  </a>
</div>

### Key Modules

Core components and utilities to build, operate, and enhance CAMEL-AI agents and societies.

| Module                                                                           | Description                                                       |
| :------------------------------------------------------------------------------- | :---------------------------------------------------------------- |
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)**                     | Core agent architectures and behaviors for autonomous operation.      |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)**             | Components for building and managing multi-agent systems.           |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)**          | Tools and methods for synthetic data creation and augmentation.  |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)**                     | Model architectures and customization options.                     |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)**                       | Tools integration for specialized agent tasks.                    |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)**                     | Memory storage and retrieval mechanisms.                          |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)**                   | Persistent storage solutions for agent data and states.          |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)**      | Performance evaluation and testing frameworks.                    |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)**           | Code and command interpretation capabilities.                   |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)**               | Data ingestion and preprocessing tools.                        |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)**              | Knowledge retrieval and RAG components.                          |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)**          | Execution environment and process management.                    |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Interactive components for human oversight and intervention.    |
---

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

>### Research with US
>
>We warmly invite you to use CAMEL for your impactful research.
>
> Rigorous research takes time and resources. We are a community-driven research collective with 100+ researchers exploring the frontier research of Multi-agent Systems. Join our ongoing projects or test new ideas with us, [reach out via email](mailto:camel-ai@eigent.ai) for more information.
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

Find practical guides for implementing CAMEL-AI agents and societies.

### 1. Basic Concepts

| Cookbook                                                                                                           | Description                                                                                                    |
| :----------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**     | A step-by-step guide to building your first agent.                                                             |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)** | Learn to build a collaborative society of agents.                                                            |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)**                   | Best practices for message handling in agents.                                                                |

### 2. Advanced Features

| Cookbook                                                                                                             | Description                                                                                                |
| :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)**                  | Integrating tools for enhanced functionality.                                                              |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)**                | Implementing memory systems in agents.                                                                     |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)**                     | Recipes for Retrieval-Augmented Generation.                                                              |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)**          | Leveraging knowledge graphs with RAG.                                                                      |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)** | Tools for tracking and managing agents in operations.                                                     |

### 3. Model Training & Data Generation

| Cookbook                                                                                                                             | Description                                                                                                           |
| :------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)** | Learn how to generate data with CAMEL and fine-tune models effectively with Unsloth.                                 |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)** | Explore how to generate data with real function calls and the Hermes format.                                         |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)** | Uncover how to generate CoT data with CAMEL and seamlessly upload it to Huggingface.                                 |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)** | Discover how to generate CoT data using CAMEL and SFT Qwen with Unsolth, and seamlessly upload your data and model to Huggingface. |

### 4. Multi-Agent Systems & Applications

| Cookbook                                                                                                                     | Description                                                                                                |
| :--------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)** | Create role-playing agents for data scraping and reporting.                                                 |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)** | Building a team of agents for collaborative judging.                                                        |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** | Build dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system.  |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)** | Learn how to build a robust customer service bot for Discord using Agentic RAG.                                |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)** | Learn how to build a robust customer service bot for Discord using Agentic RAG which supports local deployment. |

### 5. Data Processing

| Cookbook                                                                                               | Description                                                                         |
| :----------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- |
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)**          | Techniques for agents in video data analysis.                                     |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Explore methods for extracting and processing data from websites using Firecrawl. |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Learn how to create AI agents that work with your PDFs.                                |

<br>

## Real-World Usecases

CAMEL empowers real business value across various applications:

### 1 Infrastructure Automation

| Usecase                                                                  | Description                                                  |
| :----------------------------------------------------------------------- | :----------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)**                                |  -  |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** | Intelligent agents manage Cloudflare resources.                |

### 2 Productivity & Business Workflows

| Usecase                                                            | Description                                                  |
| :----------------------------------------------------------------- | :----------------------------------------------------------- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)**                          | Coordinate agents to manage Airbnb listings.                |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Analyze PowerPoint documents through collaboration.         |

### 3 Retrieval-Augmented Multi-Agent Chat

| Usecase                                                                | Description                                                  |
| :--------------------------------------------------------------------- | :----------------------------------------------------------- |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)**                  | Query and understand GitHub codebases.                      |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)**                | Conversational agents extract and summarize video transcripts. |

### 4 Video & Document Intelligence

| Usecase                                                                  | Description                                                  |
| :----------------------------------------------------------------------- | :----------------------------------------------------------- |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)**                               | Agents perform OCR on video screenshots.                     |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)**                               | CAMEL agents use OCR with Mistral to analyze documents.      |

### 5 Research & Collaboration

| Usecase                                                                         | Description                                                  |
| :------------------------------------------------------------------------------ | :----------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Simulates a team of research agents collaborating on literature review. |

<br>

## 🗓️ Events

Stay connected with CAMEL through:

-   🎙️ **Community Meetings** — Weekly virtual syncs
-   🏆 **Competitions** — Hackathons and coding challenges
-   🤝 **Volunteer Activities** — Contributions, documentation drives
-   🌍 **Ambassador Programs** — Represent CAMEL

>  Join our [Discord](https://discord.com/invite/CNcNpquyDc) or [Ambassador Program](https://www.camel-ai.org/ambassador) to get involved!

<br>

## Contributing to CAMEL

>  Review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) to get started.
>
>   Share CAMEL on social media, at events, or during conferences!

<br>

## Community & Contact

*   **GitHub Issues:** [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** [Learn more](https://www.camel-ai.org/community)
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

Special thanks to [Nomic AI](https://home.nomic.ai/) and Haya Hammoud.  Also, thanks to:

*   `TaskCreationAgent`, `TaskPrioritizationAgent`, and `BabyAGI` from *Nakajima et al.*.
*   `PersonaHub` from *Tao Ge et al.*.
*   `Self-Instruct` from *Yizhong Wang et al.*.

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
[huggingface-url]: https://huggingface.co/camel-ai
[huggingface-image]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CAMEL--AI-ffc107?color=ffc107&logoColor=white
[discord-url]: https://discord.camel-ai.org/
[discord-image]: https://img.shields.io/discord/1082486657678311454?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb
[wechat-url]: https://ghli.org/camel/wechat.png
[wechat-image]: https://img.shields.io/badge/WeChat-CamelAIOrg-brightgreen?logo=wechat&logoColor=white
[x-url]: https://x.com/CamelAIOrg
[x-image]: https://img.shields.io/twitter/follow/CamelAIOrg?style=social
[twitter-image]: https://img.shields.io/twitter/follow/CamelAIOrg?style=social&color=brightgreen&logo=twitter
[reddit-url]: https://www.reddit.com/r/CamelAI/
[reddit-image]: https://img.shields.io/reddit/subreddit-subscribers/CamelAI?style=plastic&logo=reddit&label=r%2FCAMEL&labelColor=white
[ambassador-url]: https://www.camel-ai.org/community
[package-download-url]: https://pypi.org/project/camel-ai