<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/banner.png" alt="CAMEL-AI Banner">
  </a>
</div>

<div align="center">

[![Documentation][docs-image]][docs-url]
[![Discord][discord-image]][discord-url]
[![X (Twitter)][x-image]][x-url]
[![Reddit][reddit-image]][reddit-url]
[![WeChat][wechat-image]][wechat-url]
[![Hugging Face][huggingface-image]][huggingface-url]
[![GitHub Stars][star-image]][star-url]
[![Package License][package-license-image]][package-license-url]
[![PyPI Downloads][package-download-image]][package-download-url]

<a href="https://trendshift.io/repositories/649" target="_blank"><img src="https://trendshift.io/api/badge/repositories/649" alt="camel-ai/camel | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

<hr>

## CAMEL: Build and Explore Multi-Agent Systems with Cutting-Edge AI

**CAMEL is an open-source framework empowering researchers and developers to explore the scaling laws of AI agents and build innovative multi-agent systems.**  Explore the [original repo](https://github.com/camel-ai/camel).

<div align="center">
  <h4 align="center">
    <a href="https://github.com/camel-ai/camel#community">Community</a> |
    <a href="https://github.com/camel-ai/camel#installation">Installation</a> |
    <a href="https://github.com/camel-ai/camel/tree/HEAD/examples">Examples</a> |
    <a href="https://arxiv.org/abs/2303.17760">Paper</a> |
    <a href="https://github.com/camel-ai/camel#citation">Citation</a> |
    <a href="https://github.com/camel-ai/camel#contributing-to-camel-">Contributing</a> |
    <a href="https://www.camel-ai.org/">CAMEL-AI Website</a>
  </h4>
</div>

<p style="line-height: 1.5; text-align: center;">
  CAMEL facilitates groundbreaking research in multi-agent systems by providing tools to implement and support various agents, tasks, prompts, models, and simulated environments. Join our thriving community to push the boundaries of agent capabilities.
</p>

<br>

<div align="center">
  <a href="https://discord.camel-ai.org/">Join our Discord Community</a>  or scan the WeChat QR code below to connect with the community.
  <br>
  <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="150">
</div>

<div align="center">
  <img src="docs/images/star.gif" alt="Star" width="186" height="60">
</div>

<br>

## Key Features:

*   ✅ **Evolvability:**  Continuously evolve multi-agent systems through data generation and environment interactions, driven by reinforcement or supervised learning.
*   ✅ **Scalability:** Supports systems with millions of agents for efficient coordination, communication, and resource management.
*   ✅ **Statefulness:** Agents maintain stateful memory for multi-step interactions and complex task handling.
*   ✅ **Code-as-Prompt:**  Leverage code and comments as prompts for agents, promoting clear and readable code for both humans and AI.
*   ✅ **Large-Scale Agent System:** Simulate up to 1M agents to study emergent behaviors and scaling laws.
*   ✅ **Dynamic Communication:** Enable real-time interactions among agents, fostering seamless collaboration.
*   ✅ **Support for Multiple Benchmarks:** Rigorously evaluate agent performance using standardized benchmarks.
*   ✅ **Diverse Agent Types:** Work with varied agent roles, tasks, models, and environments.
*   ✅ **Data Generation and Tool Integration:** Automate large-scale dataset creation and integrate with multiple tools, streamlining research workflows.

<br>

## Why Choose CAMEL for Your Research?

CAMEL is a community-driven research collective with over 100 researchers dedicated to advancing the field of Multi-Agent Systems. Here's why researchers worldwide choose CAMEL:

*   **Large-Scale Agent System:** Simulate up to 1 million agents to study emergent behaviors and scaling laws.
*   **Dynamic Communication:** Enable real-time agent interactions for complex task collaboration.
*   **Stateful Memory:** Equip agents with historical context for improved decision-making.
*   **Multiple Benchmarks:** Utilize standardized benchmarks for reproducible and reliable comparisons.
*   **Diverse Agent Types:** Support various agent roles, tasks, models, and environments.
*   **Data Generation and Tool Integration:** Automate data creation and integrate with research tools.

<br>

## What Can You Build With CAMEL?

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

Easily install CAMEL using pip:

```bash
pip install camel-ai
```

### Getting Started: ChatAgent Example

This example demonstrates how to create a `ChatAgent` and perform a search using DuckDuckGo.

1.  **Install web tools:**

    ```bash
    pip install 'camel-ai[web_tools]'
    ```

2.  **Set your OpenAI API key:**

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

For detailed instructions, explore the [installation guide](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).
Build multi-agent systems with our CAMEL Tech Stack and Cookbooks at [docs.camel-ai.org](https://docs.camel-ai.org).

Explore these resources to get started:

*   **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**
*   **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)**
*   **[Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)**
*   **[Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)**

##  Tech Stack

<div align="center">
  <a href="https://docs.camel-ai.org">
    <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="CAMEL Tech Stack">
  </a>
</div>

### Key Modules

Core components and utilities for building, operating, and enhancing CAMEL-AI agents and societies.

| Module                                        | Description                                                                        |
| :--------------------------------------------- | :--------------------------------------------------------------------------------- |
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)**                     | Core agent architectures and behaviors.                                     |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)**               | Multi-agent system and collaboration management.                            |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)**             | Tools for synthetic data creation.                                           |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)**                      | Model architectures and customization.                                     |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)**                       | Tools integration for specialized tasks.                                     |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)**                      | Agent state management.                                                      |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)**                    | Persistent storage for agent data.                                             |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)** | Performance evaluation frameworks.                                          |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)**        | Code and command interpretation.                                               |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)**                | Data ingestion and preprocessing.                                           |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)**               | Knowledge retrieval and RAG components.                                       |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)**            | Execution environment and process management.                                  |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Interactive components for human oversight and intervention. |

---

## Research

Explore our active research projects:

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
>
>Join our ongoing projects or test your ideas. [Contact us](mailto:camel-ai@eigent.ai) for more information.
>
><div align="center">
>    <img src="docs/images/partners.png" alt="Partners">
></div>

<br>

## Synthetic Datasets

### 1. LLM Backend Support

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

## Cookbooks (Use Cases)

Practical guides and tutorials for implementing specific functionalities.

### 1. Basic Concepts

| Cookbook                                                                                                                                      | Description                                            |
| :--------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- |
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**                          | Build your first agent.                             |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)**          | Build a collaborative society of agents.               |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)**                                                 | Message handling best practices.                      |

### 2. Advanced Features

| Cookbook                                                                                                                                        | Description                                                                         |
| :----------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- |
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)**                                            | Integrating tools for functionality.                                                |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)**                                          | Implementing memory systems.                                                      |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)**                                               | Recipes for Retrieval-Augmented Generation.                                         |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)**                                     | Leveraging knowledge graphs with RAG.                                              |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)** | Tools for tracking and managing agents in operations.                         |

### 3. Model Training & Data Generation

| Cookbook                                                                                                                                                                           | Description                                                                                                                            |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------- |
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)**          | Generate data and fine-tune models with Unsloth.                                                                                     |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)**              | Generate data using real function calls and the Hermes format.                                                                        |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)** | Generate CoT data and upload it to Hugging Face.  |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)** | Generate CoT data, SFT Qwen with Unsloth and upload to Hugging Face.  |

### 4. Multi-Agent Systems & Applications

| Cookbook                                                                                                                                                   | Description                                                                                            |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)**                 | Role-playing agents for data scraping and reporting.                                                  |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)**              | Build a team of agents for collaborative judging.                                                     |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** | Build dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system. |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)** | Build a customer service bot for Discord with Agentic RAG.                                           |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)** | Build a local customer service bot for Discord with Agentic RAG.     |

### 5. Data Processing

| Cookbook                                                                                                                                 | Description                                                 |
| :----------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)**                                              | Agent techniques in video data analysis.                    |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Extract and process data from websites using Firecrawl.      |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Build AI agents for PDF processing using Chunkr and Mistral. |

<br>

## Real-World Use Cases

Demonstrations of how CAMEL enables real business value:

### 1. Infrastructure Automation

| Use Case                                                                | Description                                                                 |
| :---------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)** | Automate infrastructure with multi-agent framework.                          |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** | Dynamically manage Cloudflare resources for efficient cloud performance. |

### 2. Productivity & Business Workflows

| Use Case                                                                | Description                                                                   |
| :---------------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)** | Optimize and manage Airbnb listings.                                       |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Extract insights from PowerPoint documents.                                |

### 3. Retrieval-Augmented Multi-Agent Chat

| Use Case                                                                | Description                                                                     |
| :---------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)** | Query and understand GitHub codebases.                                       |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)** | Extract and summarize video transcripts.                                      |

### 4. Video & Document Intelligence

| Use Case                                                                | Description                                                                      |
| :---------------------------------------------------------------------- | :------------------------------------------------------------------------------- |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | Summarize video content using OCR.                                              |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | Analyze documents using OCR with Mistral.                                       |

### 5. Research & Collaboration

| Use Case                                                                | Description                                                                      |
| :---------------------------------------------------------------------- | :------------------------------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Simulate a research team for literature review.                               |

<br>

## 🗓️ Events

*   🎙️ **Community Meetings:** Weekly virtual syncs.
*   🏆 **Competitions:** Hackathons, coding challenges.
*   🤝 **Volunteer Activities:** Contributions, documentation drives.
*   🌍 **Ambassador Programs:** Represent CAMEL in your community.

Join our [Discord](https://discord.com/invite/CNcNpquyDc) or the [Ambassador Program](https://www.camel-ai.org/community).

<br>

## Contributing

Review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) to get started.  Share CAMEL on social media!

<br>

## Community & Contact

*   **GitHub Issues:** Report bugs, request features. [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** Get real-time support, chat with the community. [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** Follow for updates. [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** [Learn more](https://www.camel-ai.org/community)
*   **WeChat Community:** Scan the QR code below:

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

Special thanks to [Nomic AI](https://home.nomic.ai/).  Thanks to Haya Hammoud for the initial logo design.  Please cite the original works if you use the modules.

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