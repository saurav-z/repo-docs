<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/banner.png" alt="CAMEL Banner">
  </a>
</div>

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

## CAMEL: The Leading Framework for Multi-Agent LLM Research

**CAMEL is an open-source framework revolutionizing the study of AI agents, providing tools and resources to explore their behavior, capabilities, and potential risks. Dive in, explore the code, and contribute to the future of AI: [Explore the CAMEL Repo](https://github.com/camel-ai/camel)**

<div align="center">
<h4 align="center">

[Community](https://github.com/camel-ai/camel#community) |
[Installation](https://github.com/camel-ai/camel#installation) |
[Examples](https://github.com/camel-ai/camel/tree/HEAD/examples) |
[Paper](https://arxiv.org/abs/2303.17760) |
[Citation](https://github.com/camel-ai/camel#citation) |
[Contributing](https://github.com/camel-ai/camel#contributing-to-camel-) |
[CAMEL-AI](https://www.camel-ai.org/)

</h4>

<p style="line-height: 1.5; text-align: center;"> 🐫 CAMEL empowers researchers to explore the scaling laws of AI agents. We provide tools and frameworks to implement and support various types of agents, tasks, prompts, models, and simulated environments.</p>

<br>

Join our community ([*Discord*](https://discord.camel-ai.org/) or [*WeChat*](https://ghli.org/camel/wechat.png)) and help push the boundaries of multi-agent systems.

🌟 Star CAMEL on GitHub to stay updated on new releases!

</div>

<div align="center">
    <img src="docs/images/star.gif" alt="Star" width="186" height="60">
  </a>
</div>

<br>

## Key Features of the CAMEL Framework

*   **Large-Scale Agent Systems:** Simulate up to millions of agents to study emergent behaviors and scaling laws.
*   **Dynamic Communication:** Enable real-time interactions among agents for seamless collaboration.
*   **Stateful Memory:** Equip agents with memory to retain and leverage historical context, improving decision-making.
*   **Code-as-Prompt:** Utilize code and comments as prompts for agents to interpret effectively.
*   **Evolvability:** Facilitate continuous evolution of multi-agent systems through data generation and environment interaction.
*   **Scalability:** Support systems with millions of agents, ensuring efficient coordination and resource management.

<br>

## Why Choose CAMEL for Your Research?

CAMEL is the premier choice for multi-agent systems research, offering:

*   A strong, community-driven research collective with over 100 researchers.
*   Standardized benchmarks for rigorous evaluation and reliable comparisons.
*   Support for diverse agent roles, tasks, models, and environments.
*   Tools for automating data generation and integrating with various research tools.

<br>

## What Can You Build with CAMEL?

CAMEL enables you to build a wide range of applications:

### 1. Data Generation

<div align="center">
  <a href="https://github.com/camel-ai/camel/blob/master/camel/datagen/cot_datagen.py">
    <img src="docs/images/cot.png" alt="CoT Data Generation">
  </a>
</div>

<div align="center">
  <a href="https://github.com/camel-ai/camel/tree/master/camel/datagen/self_instruct">
    <img src="docs/images/self-instruct.png" alt="Self-Instruct Data Generation">
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

Easily install CAMEL via PyPI with:

```bash
pip install camel-ai
```

### Starting with ChatAgent

Example showing `ChatAgent` usage with DuckDuckGo search.

1.  **Install Web Tools:**

    ```bash
    pip install 'camel-ai[web_tools]'
    ```

2.  **Set OpenAI API Key:**

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

3.  **Run Python Code:**

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

For more detailed instructions, consult the [installation section](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).

Explore the CAMEL Tech Stack and Cookbooks at [docs.camel-ai.org](https://docs.camel-ai.org) to build powerful multi-agent systems.

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing) - Colab demo.

Explore various agent types:

*   **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**
*   **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)**
*   **[Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)**
*   **[Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)**

### Need Help?

Contact us on [CAMEL Discord](https://discord.camel-ai.org/) for assistance.

<br>

## Tech Stack

<div align="center">
  <a href="https://docs.camel-ai.org">
    <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="TechStack">
  </a>
</div>

### Key Modules

Core components for building and operating CAMEL-AI agents and societies.

| Module | Description |
|:---|:---|
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)** | Core agent architectures and behaviors. |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)** | Multi-agent system management. |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)** | Synthetic data creation. |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)** | Model architectures and customization. |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)** | Tool integration. |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)** | Agent state management. |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)** | Persistent storage solutions. |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)** | Performance evaluation. |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)** | Code and command interpretation. |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)** | Data ingestion and preprocessing. |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)** | Knowledge retrieval and RAG. |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)** | Execution environment and management. |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Human oversight. |
---

## Research

We research the behaviors, capabilities, and risks of AI agents at scale.

**Explore our projects:**

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
>Join our community of 100+ researchers! Explore ongoing projects or test new ideas. [Reach out via email](mailto:camel-ai@eigent.ai).
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

Practical guides for implementing functionalities in CAMEL-AI agents and societies.

### 1. Basic Concepts

| Cookbook | Description |
|:---|:---|
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)** | Step-by-step guide to building your first agent. |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)** | Build a collaborative agent society. |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)** | Best practices for message handling. |

### 2. Advanced Features

| Cookbook | Description |
|:---|:---|
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)** | Integrating tools. |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)** | Implementing memory systems. |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)** | Recipes for Retrieval-Augmented Generation. |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)** | Leveraging knowledge graphs with RAG. |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)** | Tracking and managing agents. |

### 3. Model Training & Data Generation

| Cookbook | Description |
|:---|:---|
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)** | Generate data and fine-tune with Unsloth. |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)** | Generate data with function calls and Hermes. |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)** | Generate CoT data and upload to Huggingface. |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)** | Generate CoT data with Qwen and Unsloth. |

### 4. Multi-Agent Systems & Applications

| Cookbook | Description |
|:---|:---|
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)** | Data scraping and reporting with agents. |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)** | Building a team of agents for collaborative judging. |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** |  Builds dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system. It processes financial reports, news articles, and research papers to help traders analyze data, identify relationships, and uncover market insights. The system also utilizes diverse and optional element node deduplication techniques to ensure data integrity and optimize graph structure for financial decision-making. |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)** | Build a customer service bot for Discord using Agentic RAG. |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)** | Build a customer service bot for Discord with local deployment. |

### 5. Data Processing

| Cookbook | Description |
|:---|:---|
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)** | Agents in video data analysis. |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Extract data from websites with Firecrawl. |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Create AI agents for PDF parsing. |

<br>

## Real-World Usecases

CAMEL's framework drives real-world value across:

### 1 Infrastructure Automation

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)** | Infrastructure Automation. |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** | Dynamic management of Cloudflare resources. |

### 2 Productivity & Business Workflows

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)** | Optimize and manage Airbnb listings. |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Analyze PowerPoint documents. |

### 3 Retrieval-Augmented Multi-Agent Chat

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)** | Understand GitHub codebases. |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)** | Summarize video transcripts. |

### 4 Video & Document Intelligence

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | Summarize visual content from videos. |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | Analyze documents with Mistral OCR. |

### 5 Research & Collaboration

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Collaborate on literature review. |

<br>

## 🗓️ Events

Join our community events:

-   🎙️ **Community Meetings** — Weekly syncs.
-   🏆 **Competitions** — Hackathons and challenges.
-   🤝 **Volunteer Activities** — Contributions, documentation.
-   🌍 **Ambassador Programs** — Represent CAMEL in your area.

> Host or participate in a CAMEL event! Join our [Discord](https://discord.com/invite/CNcNpquyDc) or the [Ambassador Program](https://www.camel-ai.org/ambassador).

## Contributing to CAMEL

> Contribute code and help us grow! Review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) for a smooth collaboration.
>
> Share CAMEL on social media and at events!

<br>

## Community & Contact

*   **GitHub Issues:** Report bugs, request features. [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** Get support and chat. [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** Follow for updates. [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** Advocate for CAMEL. [Learn more](https://www.camel-ai.org/community)
*   **WeChat:** Scan the QR code:

    <div align="center">
      <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
    </div>

For more information, contact camel-ai@eigent.ai.

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

We implemented research ideas from other works; cite them if you use their modules:

*   `TaskCreationAgent`, `TaskPrioritizationAgent`, and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/). [[Example](https://github.com/camel-ai/camel/blob/master/examples/ai_society/babyagi_playing.py)]
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
```
Key improvements and explanations:

*   **SEO Optimization:**
    *   Included keywords like "AI agents," "multi-agent systems," and "LLM research" throughout the text.
    *   Used headings and subheadings to structure content for readability and SEO.
    *   Added alt text to images for improved image SEO.
*   **Concise and Engaging Hook:**  "CAMEL is an open-source framework revolutionizing the study of AI agents..." immediately grabs the reader's attention.
*   **Clear Sectioning and Readability:**
    *   Well-defined sections with headings, bullet points, and tables.
    *   More concise language to improve readability.
*   **Actionable Information:**
    *   Clear instructions on how to get started ("Quick Start" section).
    *   Encouragement to join the community ("Join our community").
    *   Call-to-actions to star the repo and contribute.
*   **Comprehensive Feature Listing:**  Highlights key features with descriptions.
*   **Focus on Benefits:** Explains *why* users should choose CAMEL for their research.
*   **Expanded Usecases:** Added more examples of the real-world usecases.
*   **Cleaned-Up Code:** Improved code formatting and spacing for clarity.
*   **Community and Contact Details:** Provided clear contact information and links.
*   **Citation and Acknowledgements:** Retained these important sections.
*   **Markdown Formatting:**  Ensured proper markdown formatting for consistent rendering on GitHub.
*   **Links Back to Original Repo:**  Included a direct link at the beginning for easy navigation.
*   **Cookbook and Usecase Organization:** Improved organization of the cookbooks and usecases sections.
*   **Dataset and Model Information:** Kept the dataset information, linking to the datasets.