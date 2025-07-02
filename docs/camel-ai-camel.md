<div align="center">
  <a href="https://github.com/camel-ai/camel">
    <img src="docs/images/banner.png" alt="CAMEL Banner">
  </a>
</div>

<br>

<!-- Badges Section -->
<div align="center">
  <!-- Documentation Badge -->
  <a href="https://camel-ai.github.io/camel/index.html">
    <img src="https://img.shields.io/badge/Documentation-EB3ECC?style=for-the-badge" alt="Documentation"/>
  </a>
  <!-- Discord Badge -->
  <a href="https://discord.camel-ai.org/">
    <img src="https://img.shields.io/discord/1082486657678311454?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb&style=for-the-badge" alt="Discord"/>
  </a>
  <!-- X (Twitter) Badge -->
  <a href="https://x.com/CamelAIOrg">
    <img src="https://img.shields.io/twitter/follow/CamelAIOrg?style=social&logo=x&style=for-the-badge" alt="X (Twitter)"/>
  </a>
  <!-- Reddit Badge -->
  <a href="https://www.reddit.com/r/CamelAI/">
    <img src="https://img.shields.io/reddit/subreddit-subscribers/CamelAI?style=plastic&logo=reddit&label=r%2FCAMEL&labelColor=white&style=for-the-badge" alt="Reddit"/>
  </a>
    <!-- Hugging Face Badge -->
  <a href="https://huggingface.co/camel-ai">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CAMEL--AI-ffc107?color=ffc107&logoColor=white&style=for-the-badge" alt="Hugging Face"/>
  </a>
  <!-- GitHub Stars Badge -->
  <a href="https://github.com/camel-ai/camel/stargazers">
    <img src="https://img.shields.io/github/stars/camel-ai/camel?label=Stars&logo=github&color=brightgreen&style=for-the-badge" alt="GitHub Stars"/>
  </a>
  <!-- License Badge -->
  <a href="https://github.com/camel-ai/camel/blob/master/licenses/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" alt="License"/>
  </a>
   <!-- PyPI Downloads Badge -->
  <a href="https://pypi.org/project/camel-ai">
      <img src="https://img.shields.io/pypi/dm/camel-ai?style=for-the-badge" alt="PyPI Downloads">
  </a>

  <br>
</div>
<br>

---

## CAMEL: The Open-Source Framework for Scaling Laws of Agents

CAMEL (Communicative Agents for "Mind" Exploration of Large Language Model Society) is an open-source framework and community dedicated to understanding and advancing multi-agent systems. Explore the scaling laws of agents with CAMEL, which provides a powerful and flexible platform for research and development.  [Explore the CAMEL Repo](https://github.com/camel-ai/camel).

<br>

**Key Features:**

*   ✅ **Large-Scale Agent System:** Simulate up to millions of agents to study emergent behaviors and scaling laws in complex environments.
*   ✅ **Dynamic Communication:** Enable real-time interactions among agents, fostering seamless collaboration.
*   ✅ **Stateful Memory:** Equip agents with memory to leverage historical context and improve decision-making.
*   ✅ **Support for Multiple Benchmarks:** Utilize standardized benchmarks for rigorous evaluation and comparison.
*   ✅ **Diverse Agent Types:** Work with various agent roles, tasks, models, and environments for interdisciplinary experiments.
*   ✅ **Data Generation and Tool Integration:** Automate data creation and seamlessly integrate with multiple tools.

<br>

## Framework Design Principles

*   **🧬 Evolvability:** The framework enables continuous evolution through data generation and environment interaction.
*   **📈 Scalability:** Designed to support systems with millions of agents, ensuring efficient coordination.
*   **💾 Statefulness:** Agents maintain stateful memory for multi-step interactions and complex tasks.
*   **📖 Code-as-Prompt:**  Code and comments are prompts, written clearly for both humans and agents.

<br>

## Why Use CAMEL?

Join a community of over 100 researchers and unlock the potential of multi-agent systems. CAMEL empowers researchers to:

*   Explore emergent behaviors in large-scale simulations.
*   Develop advanced agent communication strategies.
*   Improve agent decision-making through stateful memory.
*   Conduct rigorous evaluations using standardized benchmarks.
*   Experiment with diverse agent types and applications.
*   Streamline research workflows with automated data generation.

<br>

## What Can You Build with CAMEL?

CAMEL supports a wide range of applications across data generation, task automation, and world simulation:

### 1. Data Generation

<div align="center">
  <a href="https://github.com/camel-ai/camel/blob/master/camel/datagen/cot_datagen.py">
    <img src="docs/images/cot.png" alt="CoT Data Generation" width="200">
  </a>
  <a href="https://github.com/camel-ai/camel/tree/master/camel/datagen/self_instruct">
    <img src="docs/images/self_instruct.png" alt="Self-Instruct Data Generation" width="200">
  </a>
  <a href="https://github.com/camel-ai/camel/tree/master/camel/datagen/source2synth">
    <img src="docs/images/source2synth.png" alt="Source2Synth Data Generation" width="200">
  </a>
  <a href="https://github.com/camel-ai/camel/blob/master/camel/datagen/self_improving_cot.py">
    <img src="docs/images/self_improving.png" alt="Self-Improving Data Generation" width="200">
  </a>
</div>

### 2. Task Automation

<div align="center">
  <a href="https://github.com/camel-ai/camel/blob/master/camel/societies/role_playing.py">
    <img src="docs/images/role_playing.png" alt="Role Playing" width="200">
  </a>
  <a href="https://github.com/camel-ai/camel/tree/master/camel/societies/workforce">
    <img src="docs/images/workforce.png" alt="Workforce" width="200">
  </a>
  <a href="https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html">
    <img src="docs/images/rag_pipeline.png" alt="RAG Pipeline" width="200">
  </a>
</div>

### 3. World Simulation

<div align="center">
  <a href="https://github.com/camel-ai/oasis">
    <img src="docs/images/oasis_case.png" alt="Oasis Case" width="200">
  </a>
</div>

<br>

## Quick Start

Get started with CAMEL in minutes using `pip`:

```bash
pip install camel-ai
```

### Example: ChatAgent with DuckDuckGo Search

1.  **Install tools package:**

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

Explore our docs at [docs.camel-ai.org](https://docs.camel-ai.org) and Colab demo:  [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing) .

Explore different types of agents, their roles, and their applications.

-   **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**
-   **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)**
-   **[Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)**
-   **[Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)**

<br>

## Tech Stack

<div align="center">
  <a href="https://docs.camel-ai.org">
    <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="CAMEL Tech Stack" width="700">
  </a>
</div>

### Key Modules

Core components and utilities for building, operating, and enhancing CAMEL-AI agents and societies:

| Module                                                                                      | Description                                                         |
| :------------------------------------------------------------------------------------------ | :------------------------------------------------------------------ |
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)**                          | Core agent architectures and behaviors.                            |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)**                    | Building and managing multi-agent systems and collaboration.     |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)**                    | Synthetic data creation and augmentation tools.                   |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)**                            | Model architectures and customization for agent intelligence.      |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)**                              | Integration of tools for specialized agent tasks.                 |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)**                             | Memory storage and retrieval for agent state management.         |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)**                           | Persistent storage solutions for agent data and states.          |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)**            | Performance evaluation and testing frameworks.                   |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)**                  | Code and command interpretation capabilities.                     |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)**                      | Data ingestion and preprocessing tools.                         |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)**                    | Knowledge retrieval and RAG components.                          |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)**                  | Execution environment and process management.                    |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Interactive components for human oversight and intervention.      |

---

## Research

<div align="center">
  <a href="https://crab.camel-ai.org/">
    <img src="docs/images/crab.png" alt="CRAB" width="200">
  </a>
  <a href="https://agent-trust.camel-ai.org/">
    <img src="docs/images/agent_trust.png" alt="Agent Trust" width="200">
  </a>
  <a href="https://oasis.camel-ai.org/">
    <img src="docs/images/oasis.png" alt="OASIS" width="200">
  </a>
  <a href="https://emos-project.github.io/">
    <img src="docs/images/emos.png" alt="Emos" width="200">
  </a>
</div>

>### Research with US
>
>We invite you to contribute to impactful research with CAMEL.
>
> Join our community of 100+ researchers and explore the frontier of Multi-agent Systems.  [Reach out via email](mailto:camel-ai@eigent.ai) for more information.
>
><div align="center">
>    <img src="docs/images/partners.png" alt="Partners" width="700">
></div>

<br>

## Synthetic Datasets

Explore our curated datasets for training and evaluating multi-agent systems.

### 1. Utilize Various LLMs as Backends

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

Find step-by-step guides for implementing various features in CAMEL-AI:

### 1. Basic Concepts

| Cookbook                                                                                           | Description                                             |
| :------------------------------------------------------------------------------------------------- | :------------------------------------------------------ |
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**   | Build your first agent step-by-step.                     |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)**  | Build a collaborative society of agents.             |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)**   | Best practices for message handling in agents.        |

### 2. Advanced Features

| Cookbook                                                                                                                              | Description                                                                      |
| :------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------- |
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)**                             | Integrating tools for enhanced functionality.                                  |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)**                              | Implementing memory systems in agents.                                          |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)**                                  | Recipes for Retrieval-Augmented Generation (RAG).                            |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)**                            | Leveraging knowledge graphs with RAG.                                             |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)** | Tools for tracking and managing agents in operations.                            |

### 3. Model Training & Data Generation

| Cookbook                                                                                                                                                                 | Description                                                                                       |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)** | Generate data with CAMEL and fine-tune models with Unsloth.                                          |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)** | Generate data with real function calls and the Hermes format.                                       |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)**       | Generate CoT data with CAMEL and upload it to Huggingface.                                      |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)**        | Generate CoT data with CAMEL, SFT Qwen, and Unsloth, uploading to Huggingface.                     |

### 4. Multi-Agent Systems & Applications

| Cookbook                                                                                                                              | Description                                                                                                                                                                               |
| :------------------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)**                             | Create role-playing agents for data scraping and reporting.                                                                                                                             |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)** | Build a team of agents for collaborative judging.                                                                                                                                             |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** | Build dynamic, temporally-aware knowledge graphs for financial applications.                                                                                                                                       |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)**                  | Learn how to build a robust customer service bot for Discord using Agentic RAG.                                                                                                            |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)**                 | Learn how to build a robust customer service bot for Discord using Agentic RAG which supports local deployment.                                                                          |

### 5. Data Processing

| Cookbook                                                                                | Description                                                     |
| :-------------------------------------------------------------------------------------- | :-------------------------------------------------------------- |
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)**   | Techniques for agents in video data analysis.                     |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Explore three methods for extracting and processing data from websites using Firecrawl. |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Learn how to create AI agents that work with your PDFs.          |

<br>

## 🗓️ Events

-   🎙️ **Community Meetings** - Weekly virtual syncs with the CAMEL team.
-   🏆 **Competitions** - Hackathons, Bounty Tasks, and coding challenges.
-   🤝 **Volunteer Activities** - Contributions, documentation drives, and mentorship.
-   🌍 **Ambassador Programs** - Represent CAMEL in your university or local tech groups.

> Participate in or host a CAMEL event: [Join our Discord](https://discord.com/invite/CNcNpquyDc) or learn about the [Ambassador Program](https://www.camel-ai.org/ambassador).

<br>

## Contributing

We welcome contributions! Review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) to get started.  Your support helps CAMEL grow!

<br>

## Community & Contact

*   **GitHub Issues:** Report bugs and request features. [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** Get real-time support and chat with the community. [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** Follow for updates. [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** [Learn more](https://www.camel-ai.org/community)
*   **WeChat Community:** Scan the QR code.

    <div align="center">
        <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
    </div>
*   **Email:** For more information, contact: camel-ai@eigent.ai

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

Special thanks to [Nomic AI](https://home.nomic.ai/) for access to their Atlas tool. We also thank Haya Hammoud for the initial logo design.

We implemented research ideas from others; cite their works if you use their modules:
-   `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/). [[Example](https://github.com/camel-ai/camel/blob/master/examples/ai_society/babyagi_playing.py)]
-   `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094). [[Example](https://github.com/camel-ai/camel/blob/master/examples/personas/personas_generation.py)]
-   `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560). [[Example](https://github.com/camel-ai/camel/blob/master/examples/datagen/self_instruct/self_instruct.py)]

<br>

## License

The source code is licensed under Apache 2.0.