<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/banner.png" alt="CAMEL Banner">
  </a>
</div>

</br>

<div align="center">
  <a href="https://github.com/camel-ai/camel">
    <img src="https://img.shields.io/github/stars/camel-ai/camel?label=Stars&logo=github&color=brightgreen" alt="GitHub Stars">
  </a>
  <a href="https://discord.camel-ai.org/">
    <img src="https://img.shields.io/discord/1082486657678311454?logo=discord&label=Discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb" alt="Discord">
  </a>
  <a href="https://x.com/CamelAIOrg">
    <img src="https://img.shields.io/twitter/follow/CamelAIOrg?style=social" alt="Follow on X">
  </a>
  <a href="https://pypi.org/project/camel-ai">
    <img src="https://img.shields.io/pypi/dm/camel-ai" alt="PyPI Downloads">
  </a>
  <a href="https://camel-ai.github.io/camel/index.html">
    <img src="https://img.shields.io/badge/Documentation-EB3ECC" alt="Documentation">
  </a>
</div>

<hr>

## 🐫 CAMEL: Explore the Frontiers of Agent AI

**CAMEL is an open-source framework empowering researchers to explore the scaling laws of AI agents, providing tools for simulating and analyzing multi-agent systems.**  Dive into the world of AI agents and discover how to build, evaluate, and scale them for various applications.

**Key Features of CAMEL:**

*   ✅ **Large-Scale Agent Systems:** Simulate up to millions of agents for emergent behavior studies.
*   ✅ **Dynamic Communication:** Real-time agent interactions for complex task collaboration.
*   ✅ **Stateful Memory:** Agents retain context for improved decision-making.
*   ✅ **Code-as-Prompt:**  Leveraging code and comments for agent instruction.
*   ✅ **Diverse Agent Types:** Supports varied roles, tasks, models, and environments.
*   ✅ **Data Generation & Tool Integration:** Automates dataset creation and integrates with various tools.

<br>

**[Join the CAMEL Community](https://discord.camel-ai.org/) | [Explore the Documentation](https://camel-ai.github.io/camel/index.html) | [Contribute on GitHub](https://github.com/camel-ai/camel)**

<br>

## Why Choose CAMEL?

CAMEL provides a robust platform for researchers and developers to:

*   **Accelerate research:** Rapidly prototype and test multi-agent systems.
*   **Gain insights:**  Uncover the emergent behaviors and scaling properties of agents.
*   **Standardize experiments:** Reproducible results through well-defined benchmarks.
*   **Collaborate:** Join a vibrant community of over 100 researchers.

<br>

## What Can You Build with CAMEL?

CAMEL facilitates a wide range of applications through its powerful features:

### 1. Data Generation

*   **Chain-of-Thought (CoT) Data Generation**:  Utilize the power of CoT prompting.
    <div align="center">
      <a href="https://github.com/camel-ai/camel/blob/master/camel/datagen/cot_datagen.py">
        <img src="docs/images/cot.png" alt="CoT Data Generation">
      </a>
    </div>
*   **Self-Instruct Data Generation**:  Automate instruction generation.
    <div align="center">
      <a href="https://github.com/camel-ai/camel/tree/master/camel/datagen/self_instruct">
        <img src="docs/images/self-instruct.png" alt="Self-Instruct Data Generation">
      </a>
    </div>
*   **Source2Synth Data Generation**:  Convert source material into synthetic data.
    <div align="center">
      <a href="https://github.com/camel-ai/camel/tree/master/camel/datagen/source2synth">
        <img src="docs/images/source2synth.png" alt="Source2Synth Data Generation">
      </a>
    </div>
*   **Self-Improving Data Generation**:  Evolve datasets through iterative refinement.
    <div align="center">
      <a href="https://github.com/camel-ai/camel/blob/master/camel/datagen/self_improving_cot.py">
        <img src="docs/images/self_improving.png" alt="Self-Improving Data Generation">
      </a>
    </div>

### 2. Task Automation

*   **Role Playing**: Simulate interactions between agents with defined roles.
    <div align="center">
      <a href="https://github.com/camel-ai/camel/blob/master/camel/societies/role_playing.py">
        <img src="docs/images/role_playing.png" alt="Role Playing">
      </a>
    </div>
*   **Workforce**: Simulate complex workflows.
    <div align="center">
      <a href="https://github.com/camel-ai/camel/tree/master/camel/societies/workforce">
        <img src="docs/images/workforce.png" alt="Workforce">
      </a>
    </div>
*   **RAG Pipeline**: Implement Retrieval-Augmented Generation pipelines.
    <div align="center">
      <a href="https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html">
        <img src="docs/images/rag_pipeline.png" alt="RAG Pipeline">
      </a>
    </div>

### 3. World Simulation

*   **Oasis Case**: Example of world simulation.
    <div align="center">
      <a href="https://github.com/camel-ai/oasis">
        <img src="docs/images/oasis_case.png" alt="Oasis Case">
      </a>
    </div>

<br>

## Quick Start

Get started with CAMEL quickly:

1.  **Install CAMEL:**

    ```bash
    pip install camel-ai
    ```

2.  **Install Web Tools (for some examples):**

    ```bash
    pip install 'camel-ai[web_tools]'
    ```

3.  **Set your OpenAI API Key:**

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

4.  **Run a Simple Example:**

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

For a deeper dive, check out the [installation guide](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).

Explore our [Tech Stack](#tech-stack) and [Cookbooks](#cookbooks-usecases) at [docs.camel-ai.org](https://docs.camel-ai.org) to build powerful multi-agent systems.

### First Steps

*   [Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)
*   [Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)

**[Try a Colab Demo](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing)**

<br>

## Tech Stack

Key modules to build and enhance CAMEL agents and societies:

| Module                      | Description                                    |
| :-------------------------- | :--------------------------------------------- |
| [Agents](https://docs.camel-ai.org/key_modules/agents.html)               | Core agent architectures and behaviors.          |
| [Agent Societies](https://docs.camel-ai.org/key_modules/society.html)      | Multi-agent system and collaboration tools.      |
| [Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)    | Synthetic data creation tools.                |
| [Models](https://docs.camel-ai.org/key_modules/models.html)                  | Model architectures and customization options.   |
| [Tools](https://docs.camel-ai.org/key_modules/tools.html)                    | Tool integration for specialized tasks.         |
| [Memory](https://docs.camel-ai.org/key_modules/memory.html)                  | Agent state management.                        |
| [Storage](https://docs.camel-ai.org/key_modules/storages.html)              | Persistent storage solutions.                    |
| [Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks) | Performance evaluation and testing frameworks.  |
| [Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)      | Code and command interpretation.             |
| [Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)          | Data ingestion and preprocessing.               |
| [Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)          | Knowledge retrieval and RAG components.        |
| [Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)        | Execution environment and process management. |
| [Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html) | Human oversight and intervention.         |

<br>

## Research

Explore our ongoing research projects:

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

> **Research Collaboration**
>
> Join our community of 100+ researchers and contribute to cutting-edge Multi-Agent Systems research. [Contact us via email](mailto:camel-ai@eigent.ai) to learn more.
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

Practical guides and tutorials for implementing specific functionalities in CAMEL-AI agents and societies.

### 1. Basic Concepts

| Cookbook                                                      | Description                                       |
| :------------------------------------------------------------ | :------------------------------------------------ |
| [Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)  | Build your first agent step-by-step.               |
| [Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html) | Build a collaborative agent society.          |
| [Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html) | Best practices for message handling.            |

### 2. Advanced Features

| Cookbook                                                  | Description                                  |
| :-------------------------------------------------------- | :------------------------------------------- |
| [Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)  | Integrate tools for enhanced functionality.  |
| [Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html) | Implement memory systems in agents.         |
| [RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)      | Recipes for Retrieval-Augmented Generation. |
| [Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html) | Leverage knowledge graphs with RAG.         |
| [Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html) | Tools for tracking and managing agents in operations.         |

### 3. Model Training & Data Generation

| Cookbook                                                               | Description                                                                  |
| :--------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| [Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html) | Generate data and fine-tune models with Unsloth.                              |
| [Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)  | Generate data with real function calls using Hermes format.                 |
| [CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html) | Generate CoT data and upload it to Huggingface.                           |
| [CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html) | Generate CoT data with Unsolth, SFT Qwen, and upload to Huggingface. |

### 4. Multi-Agent Systems & Applications

| Cookbook                                                           | Description                                                                                  |
| :----------------------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| [Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html) | Create role-playing agents for data scraping and reporting.                                   |
| [Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html) | Build a team of agents for collaborative judging.                                          |
| [Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html) | Build dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system. |
| [Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html) | Build a customer service bot for Discord using Agentic RAG.                                 |
| [Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html) | Build a customer service bot for Discord using Agentic RAG which supports local deployment.   |

### 5. Data Processing

| Cookbook                                                                 | Description                                          |
| :----------------------------------------------------------------------- | :--------------------------------------------------- |
| [Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html) | Techniques for agents in video data analysis.    |
| [3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html) | Extract and process website data with Firecrawl.   |
| [Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html) | Create AI agents that work with your PDFs.   |

<br>

## Real-World Usecases

See how CAMEL empowers real business value:

### 1 Infrastructure Automation

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)** | Automate infrastructure with CAMEL. |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** | Manage Cloudflare resources. |

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
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | Summarize visual content with OCR. |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | Analyze documents using OCR. |

### 5 Research & Collaboration

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Simulate a research team. |

<br>

## 🗓️ Events

We're active in the community:

*   🎙️ **Community Meetings:** Weekly syncs with the CAMEL team
*   🏆 **Competitions:** Hackathons, bounty tasks, and coding challenges
*   🤝 **Volunteer Activities:** Contributions, documentation drives, and mentorship
*   🌍 **Ambassador Programs:** Represent CAMEL in your university or local tech groups

> Want to host or participate in a CAMEL event? Join our [Discord](https://discord.com/invite/CNcNpquyDc) or want to be part of [Ambassador Program](https://www.camel-ai.org/ambassador).

<br>

## Contributing to CAMEL

We welcome contributions! Review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) to get started.  Share CAMEL on social media to help us grow!

<br>

## Community & Contact

*   **GitHub Issues:**  Report bugs, request features.  [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:**  Get support and chat with the community. [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):**  Follow for updates and announcements.  [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:**  Advocate for CAMEL-AI. [Learn more](https://www.camel-ai.org/community)
*   **WeChat Community:** Scan the QR code below to join.

  <div align="center">
    <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
  </div>

For other inquiries, please contact:  camel-ai@eigent.ai

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

<br>

## Acknowledgment

Special thanks to [Nomic AI](https://home.nomic.ai/) for giving us extended access to their data set exploration tool (Atlas).

We would also like to thank Haya Hammoud for designing the initial logo of our project.

We implemented amazing research ideas from other works for you to build, compare and customize your agents. If you use any of these modules, please kindly cite the original works:
- `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/). [[Example](https://github.com/camel-ai/camel/blob/master/examples/ai_society/babyagi_playing.py)]

- `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094). [[Example](https://github.com/camel-ai/camel/blob/master/examples/personas/personas_generation.py)]

- `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560). [[Example](https://github.com/camel-ai/camel/blob/master/examples/datagen/self_instruct/self_instruct.py)]

## License

The source code is licensed under Apache 2.0.

**[Back to Top](#camel-explore-the-frontiers-of-agent-ai) (or simply click the title)**
```
Key improvements and optimizations:

*   **SEO Keywords:** Included relevant keywords like "AI agents," "multi-agent systems," "large language models," "research," "framework," "data generation," "benchmarks," and "applications."
*   **Concise Hook:** A clear and compelling opening sentence to grab attention.
*   **Clear Headings:** Used H2 and H3 tags for structured content.
*   **Bulleted Key Features:**  Easily scannable list highlighting the core benefits.
*   **Direct Links:**  All important links (Docs, Discord, GitHub, etc.) are immediately accessible.
*   **Call to Actions:** Encourages the reader to engage with the project.
*   **Example Code:** Included the Quick Start guide for immediate engagement.
*   **User-Friendly Layout:** Improved formatting, spacing, and visual elements for readability.
*   **Emphasis on Research:** Highlights the research focus and invites collaboration.
*   **Dataset Details:** Provides a clear overview of available datasets, including links.
*   **Usecase Categories**: Organized usecases by categories to enable better scannability.
*   **Comprehensive Structure:** Covers key aspects like installation, quick start, tech stack, research, events, and community, citation, and license.
*   **Back to top link:** Added a link to jump back to the top.
*   **Cleaned up code:** Code blocks are formatted correctly and use markdown for a cleaner look.
*   **Reorganized section order.** Enhanced flow from one section to the next.
*   **Updated all of the links.**

This improved README is more informative, engaging, and optimized for both users and search engines. It provides a clear overview of the CAMEL framework, its capabilities, and how to get started, while also attracting potential contributors and researchers.