<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/banner.png" alt="CAMEL Banner">
  </a>
</div>

<br>

<div align="center">
  <a href="https://github.com/camel-ai/camel">
    <img src="https://img.shields.io/github/stars/camel-ai/camel?label=Stars&logo=github&color=brightgreen" alt="GitHub Stars">
  </a>
  <a href="https://discord.camel-ai.org/">
    <img src="https://img.shields.io/discord/1082486657678311454?logo=discord&label=Discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb" alt="Discord">
  </a>
  <a href="https://twitter.com/CamelAIOrg">
    <img src="https://img.shields.io/twitter/follow/CamelAIOrg?style=social&color=brightgreen&logo=twitter" alt="Twitter">
  </a>
  <a href="https://pypi.org/project/camel-ai">
    <img src="https://img.shields.io/pypi/dm/camel-ai" alt="PyPI Downloads">
  </a>
  <a href="https://github.com/camel-ai/camel/blob/master/licenses/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
  </a>
  <a href="https://camel-ai.github.io/camel/index.html">
    <img src="https://img.shields.io/badge/Documentation-EB3ECC" alt="Documentation">
  </a>
</div>

<hr>

## CAMEL: Explore the Future of AI Agents with an Open-Source Framework

[CAMEL](https://github.com/camel-ai/camel) is a powerful, open-source framework and community dedicated to advancing research on multi-agent systems and scaling laws of agents.  Join us in pushing the boundaries of AI agent capabilities!

<br>

### Key Features

*   **Large-Scale Agent Systems:** Simulate up to 1 million agents for exploring emergent behaviors and scaling laws.
*   **Dynamic Communication:** Enable real-time interactions among agents, fostering seamless collaboration.
*   **Stateful Memory:** Equip agents with the ability to retain and leverage historical context.
*   **Code-as-Prompt:** Clear, readable code acts as prompts for agents, enhancing interpretability.
*   **Data Generation and Tool Integration:** Automate the creation of structured datasets and streamline research workflows.
*   **Diverse Agent Support:** Work with various agent roles, tasks, models, and environments.
*   **Community Driven:** Benefit from a collaborative research community with over 100 researchers.

<br>

## Why Use CAMEL?

CAMEL provides a robust platform for researchers and developers to:

*   **Conduct Large-Scale Experiments:** Study the behavior and scalability of multi-agent systems.
*   **Develop and Test Innovative Agent Architectures:** Explore various agent types and functionalities.
*   **Generate Synthetic Data:**  Create structured datasets for training and evaluating agents.
*   **Build Practical Applications:** Automate tasks and simulate complex real-world scenarios.

<br>

## What Can You Build With CAMEL?

CAMEL empowers you to create:

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

Get started with CAMEL in minutes!

1.  **Install:**

    ```bash
    pip install camel-ai
    pip install 'camel-ai[web_tools]'  # For tools like search
    ```

2.  **Set OpenAI API Key:**

    ```bash
    export OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
    ```

3.  **Run a simple example:**

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

4.  **Explore:** See the full [installation guide](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md) and [docs.camel-ai.org](https://docs.camel-ai.org) for detailed guides.

<br>

## Tech Stack

Explore the key modules that power CAMEL:

<div align="center">
  <a href="https://docs.camel-ai.org">
    <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="TechStack">
  </a>
</div>

### Key Modules

| Module                             | Description                                          |
| :--------------------------------- | :--------------------------------------------------- |
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)**       | Core agent architectures and behaviors.        |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)**  | Multi-agent system and collaboration management. |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)** | Synthetic data creation tools.               |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)**       | Agent intelligence customization.            |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)**         | Tool integration for specialized tasks.    |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)**       | Agent state management.                      |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)**     | Persistent storage solutions.                |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)** | Performance evaluation frameworks.           |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)** | Code and command interpretation.            |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)**   | Data ingestion and preprocessing.          |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)**  | Knowledge retrieval and RAG.                 |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)** | Execution environment and process management. |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Interactive components.                     |

<br>

## Research & Datasets

*   **Research Projects:**  Explore our ongoing research:
    *   [CRAB](https://crab.camel-ai.org/)
    *   [Agent Trust](https://agent-trust.camel-ai.org/)
    *   [OASIS](https://oasis.camel-ai.org/)
    *   [EMOS](https://emos-project.github.io/)

*   **Synthetic Datasets:** Utilize various LLMs as backends.

    | Dataset        | Chat Format                                                                                         | Instruction Format                                                                                               | Chat Format (Translated)                                                                   |
    |----------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
    | **AI Society** | [Chat format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_chat.tar.gz) | [Instruction format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_instructions.json) | [Chat format (translated)](https://huggingface.co/datasets/camel-ai/ai_society_translated) |
    | **Code**       | [Chat format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_chat.tar.gz)             | [Instruction format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_instructions.json)             | x                                                                                          |
    | **Math**       | [Chat format](https://huggingface.co/datasets/camel-ai/math)                                        | x                                                                                                                | x                                                                                          |
    | **Physics**    | [Chat format](https://huggingface.co/datasets/camel-ai/physics)                                     | x                                                                                                                | x                                                                                          |
    | **Chemistry**  | [Chat format](https://huggingface.co/datasets/camel-ai/chemistry)                                   | x                                                                                                                | x                                                                                          |
    | **Biology**    | [Chat format](https://huggingface.co/datasets/camel-ai/biology)                                     | x                                                                                                                | x                                                                                          |

<br>

## Cookbooks (Usecases)

Explore practical guides and tutorials:

### 1. Basic Concepts

| Cookbook | Description                                   |
| :------- | :-------------------------------------------- |
| [Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html) | Build your first agent.                      |
| [Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html) | Build a collaborative society.            |
| [Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)       | Best practices for message handling.       |

### 2. Advanced Features

| Cookbook | Description                              |
| :------- | :--------------------------------------- |
| [Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)  | Integrate tools.                         |
| [Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html) | Implement memory systems.                  |
| [RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)  | Retrieval-Augmented Generation recipes. |
| [Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)  | Knowledge graphs with RAG.               |
| [Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)  | Manage agents in operations.            |

### 3. Model Training & Data Generation

| Cookbook | Description                               |
| :------- | :---------------------------------------- |
| [Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html) | Generate data and fine-tune models effectively.       |
| [Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html) | Generate data with real function calls. |
| [CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)  | Generate CoT data and upload to Huggingface. |
| [CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html) | Generate CoT data and SFT Qwen with Unsolth.  |

### 4. Multi-Agent Systems & Applications

| Cookbook | Description                                  |
| :------- | :------------------------------------------- |
| [Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html) | Role-playing agents for data scraping.          |
| [Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html) | Build a team of agents for judging.            |
| [Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html) | Build dynamic knowledge graphs.            |
| [Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html) | Customer service bot for Discord.          |
| [Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html) | Customer service bot for Discord (local).         |

### 5. Data Processing

| Cookbook | Description |
| :------- | :---------------------------------------- |
| [Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)    | Agent techniques for video analysis.  |
| [3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html) | Extract data from websites.         |
| [Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html) | Agents that work with PDFs.           |

<br>

## Real-World Usecases

Explore practical applications:

### 1 Infrastructure Automation

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)** | Infrastructure automation.  |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** | Manage Cloudflare resources.        |

### 2 Productivity & Business Workflows

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)** | Optimize Airbnb listings. |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Analyze PowerPoint documents.        |

### 3 Retrieval-Augmented Multi-Agent Chat

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)** | Query GitHub codebases. |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)** | Summarize YouTube videos.         |

### 4 Video & Document Intelligence

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | Summarize visual content. |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | Analyze documents with Mistral.        |

### 5 Research & Collaboration

| Usecase                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Collaborate on literature review.  |

<br>

## 🗓️ Events & Community

*   **Community Meetings:** Weekly virtual syncs.
*   **Competitions:** Hackathons and coding challenges.
*   **Volunteer Activities:** Contribute, document, and mentor.
*   **Ambassador Programs:** Represent CAMEL in your local groups.

Join our [Discord](https://discord.com/invite/CNcNpquyDc) or [Ambassador Program](https://www.camel-ai.org/community)!

<br>

## Contributing

We welcome contributions! See the [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) for details.

<br>

## Community & Contact

*   **GitHub Issues:**  [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** [Learn more](https://www.camel-ai.org/community)
*   **WeChat Community:** Scan the QR code below:

  <div align="center">
    <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
  </div>

*   **Email:**  [camel-ai@eigent.ai](mailto:camel-ai@eigent.ai)

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

Special thanks to [Nomic AI](https://home.nomic.ai/) for their Atlas tool.

We acknowledge the research ideas from others, please kindly cite the original works:

*   `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/).
*   `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094).
*   `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560).

<br>

## License

Apache 2.0