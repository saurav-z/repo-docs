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

# CAMEL: The Leading Open-Source Framework for Multi-Agent Systems

**CAMEL empowers researchers and developers to explore the cutting-edge world of multi-agent systems, providing the tools to simulate, study, and understand the scaling laws of intelligent agents.** [Explore the CAMEL GitHub Repository](https://github.com/camel-ai/camel).

**Key Features:**

*   ✅ **Evolvability:** Facilitates continuous evolution of multi-agent systems through data generation and environment interactions, supporting both reinforcement and supervised learning.
*   ✅ **Scalability:** Designed to efficiently manage millions of agents, enabling research into emergent behaviors and scaling laws.
*   ✅ **Statefulness:** Enables agents to maintain stateful memory, supporting complex, multi-step interactions and sophisticated task execution.
*   ✅ **Code-as-Prompt:** Leverages every line of code as a prompt, optimizing clarity and interpretability for both human developers and intelligent agents.
*   ✅ **Diverse Agent Types:** Supports a wide range of agent roles, tasks, models, and environments for interdisciplinary experiments.
*   ✅ **Data Generation and Tool Integration:** Streamlines research workflows through automated data generation and seamless tool integration.
*   ✅ **Community Driven:** Built and maintained by a global community of 100+ researchers.

<br>

## Why Choose CAMEL?

CAMEL offers a comprehensive platform for research and development in multi-agent systems, providing:

*   **Large-Scale Agent Systems:** Simulate up to 1 million agents to study emergent behaviors.
*   **Dynamic Communication:** Enable real-time collaboration among agents.
*   **Stateful Memory:** Equip agents with historical context for improved decision-making.
*   **Benchmark Support:** Utilize standardized benchmarks for reliable comparisons.
*   **Versatile Agent Types:** Support diverse experiments with varied roles, tasks, and models.
*   **Data Generation and Tool Integration:** Automate data creation and streamline workflows.

<br>

## Get Started Quickly

Install CAMEL easily with pip:

```bash
pip install camel-ai
```

### Example: ChatAgent with DuckDuckGo Search

1.  **Install Web Tools:**

    ```bash
    pip install 'camel-ai[web_tools]'
    ```

2.  **Set OpenAI API Key:**

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

3.  **Run the Python Code:**

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

Explore our [documentation](https://docs.camel-ai.org) and [Colab demo](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing) for more details.

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
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)** | Multi-agent system components and collaboration. |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)** | Synthetic data creation and augmentation. |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)** | Model architectures and customization. |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)** | Integration for specialized agent tasks. |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)** | Agent state management. |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)** | Persistent storage solutions. |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)** | Performance evaluation frameworks. |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)** | Code and command interpretation. |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)** | Data ingestion and preprocessing. |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)** | Knowledge retrieval and RAG components. |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)** | Execution environment and process management. |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Interactive components for human oversight. |

---

## Research

Explore ongoing research projects:

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
>Join our community for impactful research on Multi-agent Systems. [Contact us](mailto:camel-ai@eigent.ai) for collaboration.
>
><div align="center">
>    <img src="docs/images/partners.png" alt="Partners">
></div>

<br>

## Synthetic Datasets

### 1. LLM Backends

For more details, see our [`Models Documentation`](https://docs.camel-ai.org/key_modules/models.html#).

> **Datasets (Hosted on Hugging Face)**

| Dataset        | Chat Format                                                                                             | Instruction Format                                                                                                   | Chat Format (translated)                                                                     |
|----------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| **AI Society** | [Chat Format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_chat.tar.gz)     | [Instruction Format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_instructions.json)   | [Chat Format (translated)](https://huggingface.co/datasets/camel-ai/ai_society_translated)   |
| **Code**       | [Chat Format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_chat.tar.gz)                 | [Instruction Format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_instructions.json)               | x                                                                                            |
| **Math**       | [Chat Format](https://huggingface.co/datasets/camel-ai/math)                                            | x                                                                                                                    | x                                                                                            |
| **Physics**    | [Chat Format](https://huggingface.co/datasets/camel-ai/physics)                                         | x                                                                                                                    | x                                                                                            |
| **Chemistry**  | [Chat Format](https://huggingface.co/datasets/camel-ai/chemistry)                                       | x                                                                                                                    | x                                                                                            |
| **Biology**    | [Chat Format](https://huggingface.co/datasets/camel-ai/biology)                                         | x                                                                                                                    | x                                                                                            |

### 2. Instruction and Task Visualizations

| Dataset          | Instructions                                                                                                                                  | Tasks                                                                                                                                     |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| **AI Society**   | [Instructions](https://atlas.nomic.ai/map/3a559a06-87d0-4476-a879-962656242452/db961915-b254-48e8-8e5c-917f827b74c6)                         | [Tasks](https://atlas.nomic.ai/map/cb96f41b-a6fd-4fe4-ac40-08e101714483/ae06156c-a572-46e9-8345-ebe18586d02b)                               |
| **Code**         | [Instructions](https://atlas.nomic.ai/map/902d6ccb-0bbb-4294-83a8-1c7d2dae03c8/ace2e146-e49f-41db-a1f4-25a2c4be2457)                         | [Tasks](https://atlas.nomic.ai/map/efc38617-9180-490a-8630-43a05b35d22d/2576addf-a133-45d5-89a9-6b067b6652dd)                               |
| **Misalignment** | [Instructions](https://atlas.nomic.ai/map/5c491035-a26e-4a05-9593-82ffb2c3ab40/2bd98896-894e-4807-9ed8-a203ccb14d5e)                         | [Tasks](https://atlas.nomic.ai/map/abc357dd-9c04-4913-9541-63e259d7ac1f/825139a4-af66-427c-9d0e-f36b5492ab3f)                               |

<br>

## Cookbooks (Use Cases)

Practical guides for implementing specific functionalities:

### 1. Basic Concepts

| Cookbook | Description |
|:---|:---|
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)** | Build your first agent. |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)** | Build a collaborative society of agents. |
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
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)** | Generate data and fine-tune models with Unsloth. |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)** | Generate data with real function calls and Hermes format. |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)** | Generate CoT data and upload to Huggingface. |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)** | Generate CoT data and SFT Qwen with Unsloth. |

### 4. Multi-Agent Systems & Applications

| Cookbook | Description |
|:---|:---|
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)** | Create role-playing agents for scraping and reporting. |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)** | Build a team of agents for collaborative judging. |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** |  Builds dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system. It processes financial reports, news articles, and research papers to help traders analyze data, identify relationships, and uncover market insights. The system also utilizes diverse and optional element node deduplication techniques to ensure data integrity and optimize graph structure for financial decision-making. |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)** | Build a customer service bot for Discord with Agentic RAG. |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)** | Build a customer service bot for Discord with Agentic RAG supporting local deployment. |

### 5. Data Processing

| Cookbook | Description |
|:---|:---|
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)** | Techniques for video data analysis. |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Extract data from websites using Firecrawl. |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Create AI agents to work with PDFs using Chunkr and Mistral AI. |

<br>

## Real-World Use Cases

Demonstrations of how CAMEL enables real business value:

### 1 Infrastructure Automation

| Use Case                                                        | Description                                                  |
| :-------------------------------------------------------------- | :----------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)**   | Infrastructure automation with CAMEL's framework. |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)**   | Intelligent agents manage Cloudflare resources. |

### 2 Productivity & Business Workflows

| Use Case                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)** | Optimize and manage Airbnb listings. |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Analyze PowerPoint documents. |

### 3 Retrieval-Augmented Multi-Agent Chat

| Use Case                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)** | Query and understand GitHub codebases. |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)** | Extract and summarize video transcripts. |

### 4 Video & Document Intelligence

| Use Case                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | Summarize video content. |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | Analyze documents using Mistral OCR. |

### 5 Research & Collaboration

| Use Case                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Collaborate on literature review. |

<br>

## 🗓️ Events

Participate in our community events:

-   🎙️ **Community Meetings** — Weekly virtual syncs.
-   🏆 **Competitions** — Hackathons, coding challenges.
-   🤝 **Volunteer Activities** — Contributions, documentation.
-   🌍 **Ambassador Programs** — Represent CAMEL in your community.

> Host or participate in CAMEL events: Join our [Discord](https://discord.com/invite/CNcNpquyDc) or [Ambassador Program](https://www.camel-ai.org/ambassador).

<br>

## Contributing

Review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) to start contributing.

Share CAMEL on social media, at events, and during conferences!

<br>

## Community & Contact

*   **GitHub Issues:** Report bugs, request features. [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** Get real-time support, chat. [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** Follow for updates. [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** Advocate for CAMEL-AI. [Learn more](https://www.camel-ai.org/community)
*   **WeChat Community:** Scan the QR code below to join our WeChat community.

  <div align="center">
    <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
  </div>

For other inquiries, contact: camel-ai@eigent.ai

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

Special thanks to [Nomic AI](https://home.nomic.ai/) for their Atlas tool access.

Thanks to Haya Hammoud for the initial logo design.

Cite the original works if using these modules:

*   `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/). [[Example](https://github.com/camel-ai/camel/blob/master/examples/ai_society/babyagi_playing.py)]
*   `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094). [[Example](https://github.com/camel-ai/camel/blob/master/examples/personas/personas_generation.py)]
*   `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560). [[Example](https://github.com/camel-ai/camel/blob/master/examples/datagen/self_instruct/self_instruct.py)]

## License

Licensed under Apache 2.0.