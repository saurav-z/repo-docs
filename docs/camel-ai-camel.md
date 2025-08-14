<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/banner.png" alt="CAMEL AI Banner">
  </a>
</div>

<div align="center">
    <a href="https://github.com/camel-ai/camel">
      <img src="docs/images/star.gif" alt="Star" width="186" height="60">
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

## CAMEL: Unleash the Power of Multi-Agent Systems for AI Research 

**CAMEL** is an open-source framework designed for building, experimenting with, and scaling multi-agent systems, providing valuable insights into agent behaviors and scaling laws, with the goal of advancing AI research.  [Explore the CAMEL GitHub Repository](https://github.com/camel-ai/camel).

### Key Features:

*   ✅ **Large-Scale Agent System:** Simulate up to 1 million agents to study emergent behaviors and scaling laws.
*   ✅ **Dynamic Communication:** Enable real-time interactions for seamless collaboration.
*   ✅ **Stateful Memory:** Equip agents with historical context for improved decision-making.
*   ✅ **Multiple Benchmark Support:** Evaluate agent performance with standardized benchmarks.
*   ✅ **Diverse Agent Types:** Work with a variety of roles, tasks, models, and environments.
*   ✅ **Data Generation & Tool Integration:** Automate data creation and integrate with various tools.

### Why Choose CAMEL?

CAMEL is a community-driven research initiative supported by over 100 researchers, providing a comprehensive platform for exploring the frontiers of multi-agent systems.

<table style="width: 100%;">
  <tr>
    <td align="left"></td>
    <td align="left" style="font-weight: bold;">Benefits</td>
    <td align="left"></td>
  </tr>
  <tr>
    <td align="left">✅</td>
    <td align="left">Cutting-Edge Research</td>
    <td align="left">Join a community focused on the scaling laws of agents.</td>
  </tr>
  <tr>
    <td align="left">✅</td>
    <td align="left">Rapid Prototyping</td>
    <td align="left">Easily create and test diverse agent behaviors and interactions.</td>
  </tr>
  <tr>
    <td align="left">✅</td>
    <td align="left">Reproducibility</td>
    <td align="left">Standardized benchmarks and data for consistent results.</td>
  </tr>
</table>

## Getting Started

### Installation

Install CAMEL easily using pip:

```bash
pip install camel-ai
```

For more detailed instructions, visit the [installation guide](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).

### Example: ChatAgent

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

Explore our CAMEL Tech Stack and Cookbooks at [docs.camel-ai.org](https://docs.camel-ai.org) to build powerful multi-agent systems.

We provide a [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing) demo showcasing a conversation between two ChatGPT agents playing roles as a python programmer and a stock trader collaborating on developing a trading bot for stock market.

Explore different types of agents, their roles, and their applications.

-   [Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)
-   [Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)
-   [Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)
-   [Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)

### Seeking Help

Please reach out to us on [CAMEL discord](https://discord.camel-ai.org/) if you encounter any issue set up CAMEL.

## Core Modules & Tech Stack

The CAMEL framework is built on several key modules, providing a comprehensive toolkit for research and development in multi-agent systems.

<div align="center">
  <a href="https://docs.camel-ai.org">
    <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="TechStack">
  </a>
</div>

**Key Modules**

| Module | Description |
|---|---|
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)** | Core agent architectures and behaviors for autonomous operation. |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)** | Components for building and managing multi-agent systems and collaboration. |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)** | Tools and methods for synthetic data creation and augmentation. |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)** | Model architectures and customization options for agent intelligence. |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)** | Tools integration for specialized agent tasks. |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)** | Memory storage and retrieval mechanisms for agent state management. |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)** | Persistent storage solutions for agent data and states. |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)** | Performance evaluation and testing frameworks. |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)** | Code and command interpretation capabilities. |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)** | Data ingestion and preprocessing tools. |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)** | Knowledge retrieval and RAG components. |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)** | Execution environment and process management. |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Interactive components for human oversight and intervention. |
---

## Research & Projects

CAMEL drives cutting-edge research in multi-agent systems.

**Explore our Research Projects:**

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

## Cookbooks

Explore practical guides and tutorials for implementing specific functionalities.

### 1. Basic Concepts

| Cookbook | Description |
|---|---|
| [Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html) | A step-by-step guide to building your first agent. |
| [Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html) | Learn to build a collaborative society of agents. |
| [Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html) | Best practices for message handling in agents. |

### 2. Advanced Features

| Cookbook | Description |
|---|---|
| [Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html) | Integrating tools for enhanced functionality. |
| [Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html) | Implementing memory systems in agents. |
| [RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html) | Recipes for Retrieval-Augmented Generation. |
| [Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html) | Leveraging knowledge graphs with RAG. |
| [Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html) | Tools for tracking and managing agents in operations. |

### 3. Model Training & Data Generation

| Cookbook | Description |
|---|---|
| [Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html) | Learn how to generate data with CAMEL and fine-tune models effectively with Unsloth. |
| [Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html) | Explore how to generate data with real function calls and the Hermes format. |
| [CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html) | Uncover how to generate CoT data with CAMEL and seamlessly upload it to Huggingface. |
| [CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html) | Discover how to generate CoT data using CAMEL and SFT Qwen with Unsolth, and seamlessly upload your data and model to Huggingface. |

### 4. Multi-Agent Systems & Applications

| Cookbook | Description |
|---|---|
| [Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html) | Create role-playing agents for data scraping and reporting. |
| [Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html) | Building a team of agents for collaborative judging. |
| [Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html) |  Builds dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system. It processes financial reports, news articles, and research papers to help traders analyze data, identify relationships, and uncover market insights. The system also utilizes diverse and optional element node deduplication techniques to ensure data integrity and optimize graph structure for financial decision-making. |
| [Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html) | Learn how to build a robust customer service bot for Discord using Agentic RAG. |
| [Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html) | Learn how to build a robust customer service bot for Discord using Agentic RAG which supports local deployment. |

### 5. Data Processing

| Cookbook | Description |
|---|---|
| [Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html) | Techniques for agents in video data analysis. |
| [3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html) | Explore three methods for extracting and processing data from websites using Firecrawl. |
| [Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html) | Learn how to create AI agents that work with your PDFs using Chunkr and Mistral AI. |

## Real-World Usecases

CAMEL's multi-agent framework empowers real-world business value across diverse applications.

### 1 Infrastructure Automation

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp) | Intelligent agents manage and optimize ACI resources. |
| [Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel) | Agents for cloud security and performance tuning. |

### 2 Productivity & Business Workflows

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp) | Agents to optimize and manage Airbnb listings and host operations. |
| [PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase) | Analyze PowerPoint documents and extract insights. |

### 3 Retrieval-Augmented Multi-Agent Chat

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github) | Query GitHub codebases through RAG-style workflows. |
| [Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube) | Conversational agents extract and summarize video transcripts. |

### 4 Video & Document Intelligence

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr) | Agents perform OCR on video screenshots to summarize content. |
| [Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR) | CAMEL agents use OCR with Mistral to analyze documents. |

### 5 Research & Collaboration

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant) | Simulates research agents for literature review. |

## 🗓️ Events

Stay connected with our community through:

-   🎙️ Community Meetings: Weekly virtual syncs.
-   🏆 Competitions: Hackathons and coding challenges.
-   🤝 Volunteer Activities: Contributions and mentorship.
-   🌍 Ambassador Programs: Represent CAMEL.

>   Join our [Discord](https://discord.com/invite/CNcNpquyDc) or the [Ambassador Program](https://www.camel-ai.org/ambassador).

## Contributing

We welcome contributions! Review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md).

## Community & Contact

*   **GitHub Issues:** Report bugs and request features. [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** Get real-time support and chat. [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** Follow for updates. [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** Advocate for CAMEL. [Learn more](https://www.camel-ai.org/community)
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

Special thanks to [Nomic AI](https://home.nomic.ai/) for giving us extended access to their data set exploration tool (Atlas).

We would also like to thank Haya Hammoud for designing the initial logo of our project.

We implemented amazing research ideas from other works for you to build, compare and customize your agents. If you use any of these modules, please kindly cite the original works:
-   `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/). [[Example](https://github.com/camel-ai/camel/blob/master/examples/ai_society/babyagi_playing.py)]

-   `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094). [[Example](https://github.com/camel-ai/camel/blob/master/examples/personas/personas_generation.py)]

-   `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560). [[Example](https://github.com/camel-ai/camel/blob/master/examples/datagen/self_instruct/self_instruct.py)]

## License

Licensed under the Apache 2.0 License.