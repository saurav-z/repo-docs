<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/banner.png" alt="CAMEL AI Banner">
  </a>
</div>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-EB3ECC)](https://camel-ai.github.io/camel/index.html)
[![Discord](https://img.shields.io/discord/1082486657678311454?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb)](https://discord.camel-ai.org/)
[![X](https://img.shields.io/twitter/follow/CamelAIOrg?style=social)](https://x.com/CamelAIOrg)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/CamelAI?style=plastic&logo=reddit&label=r%2FCAMEL&labelColor=white)](https://www.reddit.com/r/CamelAI/)
[![Wechat](https://img.shields.io/badge/WeChat-CamelAIOrg-brightgreen?logo=wechat&logoColor=white)](https://ghli.org/camel/wechat.png)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CAMEL--AI-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/camel-ai)
[![Star](https://img.shields.io/github/stars/camel-ai/camel?label=stars&logo=github&color=brightgreen)](https://github.com/camel-ai/camel/stargazers)
[![Package License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/camel-ai/camel/blob/master/licenses/LICENSE)
[![PyPI Download](https://img.shields.io/pypi/dm/camel-ai)](https://pypi.org/project/camel-ai)

<a href="https://trendshift.io/repositories/649" target="_blank"><img src="https://trendshift.io/api/badge/repositories/649" alt="camel-ai/camel | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

<hr>

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
</div>

<p style="line-height: 1.5; text-align: center;">
CAMEL empowers researchers and developers to explore the cutting-edge world of multi-agent systems, providing a framework for building, simulating, and analyzing intelligent agents.
</p>

<br>

<div align="center">
Join our thriving community on [Discord](https://discord.camel-ai.org/) or [WeChat](https://ghli.org/camel/wechat.png) and help push the boundaries of agent-based AI research. 
<br>
<a href="https://github.com/camel-ai/camel">⭐ Star CAMEL on GitHub</a> to stay updated on the latest releases and innovations.
</div>

<div align="center">
    <img src="docs/images/star.gif" alt="Star" width="186" height="60">
</div>

<br>

## Key Features of CAMEL

*   **Large-Scale Agent Systems:** Simulate up to 1 million agents to study emergent behaviors and scaling laws.
*   **Dynamic Communication:** Enable real-time interactions and collaboration between agents.
*   **Stateful Memory:** Equip agents with memory to retain context for improved decision-making.
*   **Code-as-Prompt:** Clear and readable code serves as prompts for agent actions and learning.

<br>

## Why Choose CAMEL for Your Research?

CAMEL is a leading open-source framework for multi-agent systems, chosen by researchers worldwide for its:

*   **Large-Scale Agent System:**  Model complex systems with millions of agents.
*   **Dynamic Communication:** Facilitate real-time agent interactions.
*   **Stateful Memory:** Enable agents to remember and learn from past interactions.
*   **Multiple Benchmark Support:** Evaluate agent performance rigorously with standardized benchmarks.
*   **Diverse Agent Types:** Support for various roles, tasks, models, and environments.
*   **Data Generation & Tool Integration:** Automate data creation and streamline research workflows.

<br>

## What Can You Build with CAMEL?

CAMEL provides tools and resources for various applications:

### 1. Data Generation

[<img src="docs/images/cot.png" alt="CoT Data Generation" width="200">](https://github.com/camel-ai/camel/blob/master/camel/datagen/cot_datagen.py)
[<img src="docs/images/self_instruct.png" alt="Self-Instruct Data Generation" width="200">](https://github.com/camel-ai/camel/tree/master/camel/datagen/self_instruct)
[<img src="docs/images/source2synth.png" alt="Source2Synth Data Generation" width="200">](https://github.com/camel-ai/camel/tree/master/camel/datagen/source2synth)
[<img src="docs/images/self_improving.png" alt="Self-Improving Data Generation" width="200">](https://github.com/camel-ai/camel/blob/master/camel/datagen/self_improving_cot.py)

### 2. Task Automation

[<img src="docs/images/role_playing.png" alt="Role Playing" width="200">](https://github.com/camel-ai/camel/blob/master/camel/societies/role_playing.py)
[<img src="docs/images/workforce.png" alt="Workforce" width="200">](https://github.com/camel-ai/camel/tree/master/camel/societies/workforce)
[<img src="docs/images/rag_pipeline.png" alt="RAG Pipeline" width="200">](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)

### 3. World Simulation

[<img src="docs/images/oasis_case.png" alt="Oasis Case" width="200">](https://github.com/camel-ai/oasis)

<br>

## Quick Start - Get Started with CAMEL

Easily install CAMEL using pip:

```bash
pip install camel-ai
```

### Example: Creating a ChatAgent

This example demonstrates creating a `ChatAgent` and using it to query DuckDuckGo.

1.  **Install necessary dependencies:**

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
    # CAMEL-AI is the first LLM (Large Language Model) multi-agent framework
    # and an open-source community focused on finding the scaling laws of agents.
    # ...

    response_2 = agent.step("What is the Github link to CAMEL framework?")
    print(response_2.msgs[0].content)
    # The GitHub link to the CAMEL framework is
    # [https://github.com/camel-ai/camel](https://github.com/camel-ai/camel).
    ```

Find more details in the [installation section](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).

Explore CAMEL's capabilities by diving into our [Cookbooks](https://docs.camel-ai.org) and the [CAMEL Tech Stack](https://docs.camel-ai.org/key_modules/agents.html).

<a href="https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
</a>
Check out our Google Colab demo showcasing a conversation between two ChatGPT agents.

*   **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**
*   **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)**
*   **[Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)**
*   **[Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)**

### Need Help?

Reach out to the community on [CAMEL discord](https://discord.camel-ai.org/) for support.

<br>

## Tech Stack

[<img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="TechStack" width="600">](https://docs.camel-ai.org)

### Core Modules
Key components and utilities to build, operate, and enhance CAMEL-AI agents and societies.

| Module | Description |
|:---|:---|
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

## Research

We believe that studying these agents on a large scale offers valuable insights into their behaviors, capabilities, and potential risks.

**Explore our research projects:**

[<img src="docs/images/crab.png" alt="CRAB" width="200">](https://crab.camel-ai.org/)
[<img src="docs/images/agent_trust.png" alt="Agent Trust" width="200">](https://agent-trust.camel-ai.org/)
[<img src="docs/images/oasis.png" alt="OASIS" width="200">](https://oasis.camel-ai.org/)
[<img src="docs/images/emos.png" alt="Emos" width="200">](https://emos-project.github.io/)

>### Research with Us
>
>We encourage you to leverage CAMEL for impactful research!  Join our community of 100+ researchers and contribute to frontier research in Multi-agent Systems. Reach out via [email](mailto:camel-ai@eigent.ai) to learn about joining our ongoing projects or testing your own ideas.
>
><div align="center">
>    <img src="docs/images/partners.png" alt="Partners" width="400">
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

## Cookbooks (Use Cases)
Explore practical guides and tutorials for implementing specific functionalities in CAMEL-AI agents and societies.

### 1. Basic Concepts
| Cookbook | Description |
|:---|:---|
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)** | A step-by-step guide to building your first agent. |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)** | Learn to build a collaborative society of agents. |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)** | Best practices for message handling in agents. |

### 2. Advanced Features
| Cookbook | Description |
|:---|:---|
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)** | Integrating tools for enhanced functionality. |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)** | Implementing memory systems in agents. |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)** | Recipes for Retrieval-Augmented Generation. |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)** | Leveraging knowledge graphs with RAG. |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)** | Tools for tracking and managing agents in operations. |

### 3. Model Training & Data Generation
| Cookbook | Description |
|:---|:---|
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)** | Learn how to generate data with CAMEL and fine-tune models effectively with Unsloth. |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)** | Explore how to generate data with real function calls and the Hermes format. |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)** | Uncover how to generate CoT data with CAMEL and seamlessly upload it to Huggingface. |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)** | Discover how to generate CoT data using CAMEL and SFT Qwen with Unsolth, and seamlessly upload your data and model to Huggingface. |

### 4. Multi-Agent Systems & Applications
| Cookbook | Description |
|:---|:---|
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)** | Create role-playing agents for data scraping and reporting. |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)** | Building a team of agents for collaborative judging. |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** |  Builds dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system. It processes financial reports, news articles, and research papers to help traders analyze data, identify relationships, and uncover market insights. The system also utilizes diverse and optional element node deduplication techniques to ensure data integrity and optimize graph structure for financial decision-making. |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)** | Learn how to build a robust customer service bot for Discord using Agentic RAG. |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)** | Learn how to build a robust customer service bot for Discord using Agentic RAG which supports local deployment. |

### 5. Data Processing
| Cookbook | Description |
|:---|:---|
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)** | Techniques for agents in video data analysis. |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Explore three methods for extracting and processing data from websites using Firecrawl. |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Learn how to create AI agents that work with your PDFs using Chunkr and Mistral AI. |

<br>

## Real-World Usecases

Explore real-world applications that leverage CAMEL’s multi-agent framework to provide value across various domains:

### 1 Infrastructure Automation

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)** | Illustrates CAMEL's application in infrastructure automation, showcasing the power of intelligent agents in managing ACI (Application Centric Infrastructure). |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** | Showcases the dynamic management of Cloudflare resources with intelligent agents, enhancing cloud security and performance. |

### 2 Productivity & Business Workflows

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)** | Highlights the coordination of agents to manage and optimize Airbnb listings and host operations. |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Demonstrates the use of multi-agent collaboration to analyze and extract insights from PowerPoint documents. |

### 3 Retrieval-Augmented Multi-Agent Chat

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)** | Illustrates the use of CAMEL agents for querying and understanding GitHub codebases, accelerating developer onboarding and codebase navigation. |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)** | Showcases conversational agents extracting and summarizing video transcripts for faster content understanding and repurposing. |

### 4 Video & Document Intelligence

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | Shows agents performing OCR on video screenshots to summarize visual content, aiding media monitoring and compliance. |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | CAMEL agents use OCR with Mistral to analyze documents, reducing manual effort in document understanding workflows. |

### 5 Research & Collaboration

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Simulates a team of research agents collaborating on literature review, improving efficiency in exploratory analysis and reporting. |

<br>

## 🗓️ Events

Join our community events:

*   🎙️ **Community Meetings** — Weekly virtual syncs with the CAMEL team
*   🏆 **Competitions** — Hackathons, Bounty Tasks and coding challenges hosted by CAMEL
*   🤝 **Volunteer Activities** — Contributions, documentation drives, and mentorship
*   🌍 **Ambassador Programs** — Represent CAMEL in your university or local tech groups

> Want to host or participate in a CAMEL event? Join our [Discord](https://discord.com/invite/CNcNpquyDc) or apply for the [Ambassador Program](https://www.camel-ai.org/ambassador).

<br>

## Contributing to CAMEL

>  We value your contributions!  Please read our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) to begin collaborating effectively.
>
> Share CAMEL on social media to help our community grow!

<br>

## Community & Contact

*   **GitHub Issues:** [Submit an issue](https://github.com/camel-ai/camel/issues) to report bugs, request features, and track development.
*   **Discord:** Get real-time support and connect with our community: [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** Stay updated on the latest news: [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:**  [Learn more](https://www.camel-ai.org/community) and help us spread the word.
*   **WeChat Community:**

    <div align="center">
      <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
    </div>
<br>
For any questions, contact: camel-ai@eigent.ai

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

Special thanks to [Nomic AI](https://home.nomic.ai/) for providing extended access to their dataset exploration tool (Atlas).
<br>
We appreciate the research ideas from other works which we used for building, comparing, and customizing agents. If you use any of these modules, please kindly cite the original works:
<br>
-   `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/). [[Example](https://github.com/camel-ai/camel/blob/master/examples/ai_society/babyagi_playing.py)]

-   `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094). [[Example](https://github.com/camel-ai/camel/blob/master/examples/personas/personas_generation.py)]

-   `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560). [[Example](https://github.com/camel-ai/camel/blob/master/examples/datagen/self_instruct/self_instruct.py)]
<br>
## License

The source code is licensed under Apache 2.0.

```