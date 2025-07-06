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

# CAMEL: The Open-Source Framework for Multi-Agent LLM Research

CAMEL empowers researchers and developers to explore the exciting possibilities of multi-agent systems, offering a comprehensive framework for building and studying intelligent agents. Find the [original repository here](https://github.com/camel-ai/camel).

## Key Features

*   **Large-Scale Agent Systems:** Simulate up to 1 million agents to uncover emergent behaviors and scaling laws.
*   **Dynamic Communication:** Enable real-time interactions and collaboration between agents for complex task solving.
*   **Stateful Memory:** Equip agents with memory for enhanced decision-making over extended interactions.
*   **Code-as-Prompt Design:**  Leverage code and comments as prompts for agents, encouraging clear and interpretable code.
*   **Versatile Benchmarks and Agent Types:** Support for diverse agent roles, tasks, models, and environments, ideal for interdisciplinary research.
*   **Data Generation and Tool Integration:** Automate dataset creation and seamlessly integrate with tools to streamline research workflows.

## Core Design Principles

*   **Evolvability:** Continuously improve multi-agent systems through data generation and environment interaction.
*   **Scalability:**  Designed to support systems with a massive number of agents, ensuring efficient operation.
*   **Statefulness:** Agents retain memory, allowing for multi-step interactions and complex task completion.
*   **Code-as-Prompt:** Code is written for both human and agent understanding.

## Why Choose CAMEL?

CAMEL is a community-driven research platform with over 100 researchers dedicated to multi-agent systems. It offers:

*   **Comprehensive Benchmarks:** Evaluate agent performance with rigorous, reproducible benchmarks.
*   **Support for Diverse Agent Types:** Experiment with a wide array of agent roles, tasks, and environments.
*   **Streamlined Data Generation:** Automate the creation of large, structured datasets for research.
*   **Community-Driven Support:** Access to a supportive community for troubleshooting and collaboration.

## What Can You Build with CAMEL?

CAMEL allows you to build a variety of multi-agent systems for:

### Data Generation

Explore various methods for generating synthetic datasets:

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

### Task Automation

Automate tasks using multi-agent collaboration:

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

### World Simulation

Simulate complex environments for agent exploration:

<div align="center">
  <a href="https://github.com/camel-ai/oasis">
    <img src="docs/images/oasis_case.png" alt="Oasis Case">
  </a>
</div>

## Quick Start

Get started quickly with CAMEL using these steps:

1.  **Install the `camel-ai` package:**

    ```bash
    pip install camel-ai
    ```

2.  **Install the `web_tools` package (for web tools):**

    ```bash
    pip install 'camel-ai[web_tools]'
    ```

3.  **Set your OpenAI API key:**

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

4.  **Run a simple example:**

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

For more detailed information, see the [installation section](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).

## Key Modules

| Module                                                                 | Description                                                                                                                                |
| :--------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)**       | Core agent architectures and behaviors for autonomous operation.                                                                           |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)** | Components for building and managing multi-agent systems and collaboration.                                                               |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)** | Tools and methods for synthetic data creation and augmentation.                                                                             |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)**       | Model architectures and customization options for agent intelligence.                                                                     |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)**         | Tools integration for specialized agent tasks.                                                                                              |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)**       | Memory storage and retrieval mechanisms for agent state management.                                                                           |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)**     | Persistent storage solutions for agent data and states.                                                                                   |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)** | Performance evaluation and testing frameworks.                                                                                            |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)** | Code and command interpretation capabilities.                                                                                             |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)** | Data ingestion and preprocessing tools.                                                                                                  |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)** | Knowledge retrieval and RAG components.                                                                                                  |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)** | Execution environment and process management.                                                                                             |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Interactive components for human oversight and intervention.                                                                                 |

## Research

Explore cutting-edge research projects built with CAMEL:

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

### Research with Us

We invite you to use CAMEL for your research! Join our community, contribute to ongoing projects, or propose your own ideas. Contact us at [camel-ai@eigent.ai](mailto:camel-ai@eigent.ai) to learn more.

<div align="center">
    <img src="docs/images/partners.png" alt="Partners">
</div>

## Synthetic Datasets

### 1. Utilizing Various LLMs as Backends

Access datasets hosted on Hugging Face:

| Dataset        | Chat format                                                                                         | Instruction format                                                                                               | Chat format (translated)                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **AI Society** | [Chat format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_chat.tar.gz) | [Instruction format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_instructions.json) | [Chat format (translated)](https://huggingface.co/datasets/camel-ai/ai_society_translated) |
| **Code**       | [Chat format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_chat.tar.gz)             | [Instruction format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_instructions.json)             | x                                                                                          |
| **Math**       | [Chat format](https://huggingface.co/datasets/camel-ai/math)                                        | x                                                                                                                | x                                                                                          |
| **Physics**    | [Chat format](https://huggingface.co/datasets/camel-ai/physics)                                     | x                                                                                                                | x                                                                                          |
| **Chemistry**  | [Chat format](https://huggingface.co/datasets/camel-ai/chemistry)                                   | x                                                                                                                | x                                                                                          |
| **Biology**    | [Chat format](https://huggingface.co/datasets/camel-ai/biology)                                     | x                                                                                                                | x                                                                                          |

### 2. Visualizations of Instructions and Tasks

Visualize instructions and tasks for datasets:

| Dataset          | Instructions                                                                                                         | Tasks                                                                                                         |
|------------------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **AI Society**   | [Instructions](https://atlas.nomic.ai/map/3a559a06-87d0-4476-a879-962656242452/db961915-b254-48e8-8e5c-917f827b74c6) | [Tasks](https://atlas.nomic.ai/map/cb96f41b-a6fd-4fe4-ac40-08e101714483/ae06156c-a572-46e9-8345-ebe18586d02b) |
| **Code**         | [Instructions](https://atlas.nomic.ai/map/902d6ccb-0bbb-4294-83a8-1c7d2dae03c8/ace2e146-e49f-41db-a1f4-25a2c4be2457) | [Tasks](https://atlas.nomic.ai/map/efc38617-9180-490a-8630-43a05b35d22d/2576addf-a133-45d5-89a9-6b067b6652dd) |
| **Misalignment** | [Instructions](https://atlas.nomic.ai/map/5c491035-a26e-4a05-9593-82ffb2c3ab40/2bd98896-894e-4807-9ed8-a203ccb14d5e) | [Tasks](https://atlas.nomic.ai/map/abc357dd-9c04-4913-9541-63e259d7ac1f/825139a4-af66-427c-9d0e-f36b5492ab3f) |

## Cookbooks (Use Cases)

Explore practical guides and tutorials for implementing specific functionalities:

### Basic Concepts

| Cookbook                                                                                                      | Description                                          |
| :------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------- |
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)** | Build your first agent step-by-step.               |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)** | Build a collaborative agent society.          |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)**  | Best practices for message handling in agents. |

### Advanced Features

| Cookbook                                                                                              | Description                                      |
| :---------------------------------------------------------------------------------------------------- | :----------------------------------------------- |
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)**   | Integrate tools for enhanced functionality.     |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)**  | Implement memory systems in agents.             |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)**        | Recipes for Retrieval-Augmented Generation.     |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)** | Leverage knowledge graphs with RAG.              |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)**| Tools for tracking and managing agents in operations.             |

### Model Training & Data Generation

| Cookbook                                                                                                                                                                                 | Description                                                                                                                                                                                                                                        |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)**          | Generate data with CAMEL and fine-tune models using Unsloth.                                                                                                                                                                                     |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)**                   | Generate data with real function calls and the Hermes format.                                                                                                                                                                                  |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)**                           | Generate CoT data with CAMEL and upload it to Hugging Face.                                                                                                                                                                                     |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)**                             | Generate CoT data with CAMEL, SFT Qwen with Unsloth, and seamlessly upload your data and model to Hugging Face.                                                                                                                                    |

### Multi-Agent Systems & Applications

| Cookbook                                                                                                                                                                                                                | Description                                                                                                                                                        |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)**                                                                              | Create role-playing agents for data scraping and reporting.                                                                                                     |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)**                                                                        | Build a team of agents for collaborative judging.                                                                                                                 |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)**                          |  Builds dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system. It processes financial reports, news articles, and research papers to help traders analyze data, identify relationships, and uncover market insights. The system also utilizes diverse and optional element node deduplication techniques to ensure data integrity and optimize graph structure for financial decision-making. |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)**                                                | Build a customer service bot for Discord using Agentic RAG.                                                                                                         |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)**                                              | Build a customer service bot for Discord using Agentic RAG supporting local deployment.                                                                                    |

### Data Processing

| Cookbook                                                                                                   | Description                                                   |
| :--------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------ |
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)**          | Explore agent techniques for video data analysis.             |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Extract and process data from websites with Firecrawl.      |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Create AI agents that work with your PDFs using Chunkr.    |

## Real-World Use Cases

Explore how CAMEL is used to create business value:

### Infrastructure Automation

| Use Case                                                             | Description                                                                                                                                           |
| :------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)** | Automate infrastructure tasks within the Cisco ACI (Application Centric Infrastructure) environment.                                               |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** | Manage Cloudflare resources dynamically, improving cloud security and performance.                                                                 |

### Productivity & Business Workflows

| Use Case                                                            | Description                                                                                                                      |
| :------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)** | Optimize and manage Airbnb listings and host operations.                                                                        |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Analyze PowerPoint documents and extract insights through multi-agent collaboration.                                          |

### Retrieval-Augmented Multi-Agent Chat

| Use Case                                                           | Description                                                                                                                  |
| :----------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)** | Query and understand GitHub codebases, accelerating developer onboarding and codebase navigation.                             |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)** | Extract and summarize video transcripts, enabling faster content understanding and repurposing.                              |

### Video & Document Intelligence

| Use Case                                                           | Description                                                                                                  |
| :----------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | Summarize visual content from video screenshots, supporting media monitoring and compliance.                |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | Analyze documents using OCR, reducing manual effort in document understanding workflows.                       |

### Research & Collaboration

| Use Case                                                                 | Description                                                                                                                        |
| :----------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Simulate a team of research agents collaborating on literature review, improving efficiency in exploratory analysis and reporting. |

## Events

Stay connected with the CAMEL community:

*   **Community Meetings:** Weekly virtual syncs.
*   **Competitions:** Hackathons and coding challenges.
*   **Volunteer Activities:** Contribute to documentation and mentorship.
*   **Ambassador Programs:** Represent CAMEL in your university or tech group.

Join our [Discord](https://discord.com/invite/CNcNpquyDc) to host or participate in events or learn more about the [Ambassador Program](https://www.camel-ai.org/community).

## Contributing

We welcome contributions! Please review the [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) to get started.

## Community & Contact

*   **GitHub Issues:** [Submit an issue](https://github.com/camel-ai/camel/issues) to report bugs and request features.
*   **Discord:** [Join us](https://discord.camel-ai.org/) for real-time support.
*   **X (Twitter):** [Follow us](https://x.com/CamelAIOrg) for updates.
*   **Ambassador Project:** [Learn more](https://www.camel-ai.org/community) to represent CAMEL-AI.
*   **WeChat:** Scan the QR code to join our WeChat community.

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

## Acknowledgments

Special thanks to [Nomic AI](https://home.nomic.ai/) and Haya Hammoud.

Please cite the original works when using the modules we implemented.

## License

Licensed under Apache 2.0.