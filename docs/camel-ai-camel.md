<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/banner.png" alt="CAMEL Banner">
  </a>
</div>

<br>

<div align="center">

[![Documentation][docs-image]][docs-url]
[![Discord][discord-image]][discord-url]
[![X (Twitter)][x-image]][x-url]
[![Reddit][reddit-image]][reddit-url]
[![Wechat][wechat-image]][wechat-url]
[![Hugging Face][huggingface-image]][huggingface-url]
[![Star][star-image]][star-url]
[![Package License][package-license-image]][package-license-url]
[![PyPI Downloads][package-download-image]][package-download-url]

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

<p style="line-height: 1.5; text-align: center;"> 🐫 <b>CAMEL: Unleash the power of multi-agent systems!</b>  CAMEL is an open-source framework and community dedicated to researching the scaling laws of agents, providing tools and resources to build, simulate, and analyze complex multi-agent environments.  </p>

<br>

Join the CAMEL community on [Discord](https://discord.camel-ai.org/) or [WeChat](https://ghli.org/camel/wechat.png) and stay up-to-date on the latest advances!

🌟 Show your support and get notified of new releases by starring CAMEL on [GitHub](https://github.com/camel-ai/camel)!

</div>

<div align="center">
    <img src="docs/images/star.gif" alt="Star CAMEL" width="186" height="60">
</div>

<br>

## Key Features of CAMEL

*   **Large-Scale Agent Systems:** Simulate up to 1 million agents to observe emergent behaviors and scaling laws.
*   **Dynamic Communication:** Facilitate real-time interactions between agents for seamless collaboration.
*   **Stateful Memory:** Equip agents with memory for multi-step interactions and improved decision-making.
*   **Code-as-Prompt:** Leverage code and comments as prompts to guide agent behavior.
*   **Extensive Benchmarks:** Use standardized benchmarks to evaluate and compare agent performance.
*   **Diverse Agent Support:** Work with various agent roles, tasks, models, and environments for interdisciplinary research.
*   **Data Generation & Tool Integration:** Automate dataset creation and seamlessly integrate with research tools.

<br>

## Why Use CAMEL for Your Research?

CAMEL is a community-driven platform that brings together over 100 researchers, providing a robust framework for:

*   **Exploring Agent Behaviors:** Analyze how agents behave in complex environments.
*   **Advancing Multi-Agent Systems:** Contribute to cutting-edge research.
*   **Replicable Results:**  Use standardized benchmarks for robust experiments.

<br>

## What You Can Build with CAMEL

CAMEL empowers you to build powerful AI applications with these capabilities:

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

Get started with CAMEL quickly by installing it via pip:

```bash
pip install camel-ai
```

### Example: Starting with ChatAgent

This example demonstrates how to create and use a `ChatAgent`:

1.  **Install web tools:**

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

Explore our [documentation](https://docs.camel-ai.org) and [examples](https://github.com/camel-ai/camel/tree/HEAD/examples) to build your own multi-agent systems.

Check out the [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)][colab-url] demo for a conversation between two ChatGPT agents.

Discover various agent types and applications:

*   [Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)
*   [Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)
*   [Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)
*   [Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)

### Need Help?

Get support on [Discord](https://discord.camel-ai.org/) if you have any questions.

<br>

## Tech Stack

<div align="center">
  <a href="https://docs.camel-ai.org">
    <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="CAMEL Tech Stack">
  </a>
</div>

### Key Modules

Core components and utilities for building, operating, and enhancing CAMEL-AI agents and societies.

| Module                                      | Description                                                                    |
| :------------------------------------------ | :----------------------------------------------------------------------------- |
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)**                 | Core agent architectures and behaviors for autonomous operation.               |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)**            | Components for building and managing multi-agent systems and collaboration.  |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)**        | Tools and methods for synthetic data creation and augmentation.               |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)**                   | Model architectures and customization options for agent intelligence.          |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)**                    | Tools integration for specialized agent tasks.                                 |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)**                   | Memory storage and retrieval mechanisms for agent state management.           |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)**                 | Persistent storage solutions for agent data and states.                      |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)** | Performance evaluation and testing frameworks.                                 |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)**       | Code and command interpretation capabilities.                                |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)**             | Data ingestion and preprocessing tools.                                        |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)**           | Knowledge retrieval and RAG components.                                         |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)**        | Execution environment and process management.                                |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Interactive components for human oversight and intervention.                    |

---

## Research

We believe that studying agents at scale can uncover valuable insights into their behavior, capabilities, and potential risks.

**Explore our current research projects:**

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
> We invite you to use CAMEL for impactful research.
>
> Research requires time and resources. As a community-driven research collective with 100+ researchers, we are exploring the frontiers of Multi-agent Systems. Join our projects or test new ideas; [contact us via email](mailto:camel-ai@eigent.ai) for more information.
>
><div align="center">
>    <img src="docs/images/partners.png" alt="Partners">
></div>

<br>

## Synthetic Datasets

### 1.  Use Various LLMs as Backends

For details, see our [Models Documentation](https://docs.camel-ai.org/key_modules/models.html#).

> **Datasets (Hosted on Hugging Face)**

| Dataset        | Chat Format                                                                                          | Instruction Format                                                                                                | Chat Format (Translated)                                                                    |
| :------------- | :--------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------- |
| **AI Society** | [Chat format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_chat.tar.gz)  | [Instruction format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_instructions.json) | [Chat format (translated)](https://huggingface.co/datasets/camel-ai/ai_society_translated) |
| **Code**       | [Chat format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_chat.tar.gz)              | [Instruction format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_instructions.json)             | x                                                                                           |
| **Math**       | [Chat format](https://huggingface.co/datasets/camel-ai/math)                                         | x                                                                                                                 | x                                                                                           |
| **Physics**    | [Chat format](https://huggingface.co/datasets/camel-ai/physics)                                      | x                                                                                                                 | x                                                                                           |
| **Chemistry**  | [Chat format](https://huggingface.co/datasets/camel-ai/chemistry)                                    | x                                                                                                                 | x                                                                                           |
| **Biology**    | [Chat format](https://huggingface.co/datasets/camel-ai/biology)                                      | x                                                                                                                 | x                                                                                           |

### 2. Visualizations of Instructions and Tasks

| Dataset          | Instructions                                                                                                         | Tasks                                                                                                        |
| :--------------- | :---------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| **AI Society**   | [Instructions](https://atlas.nomic.ai/map/3a559a06-87d0-4476-a879-962656242452/db961915-b254-48e8-8e5c-917f827b74c6) | [Tasks](https://atlas.nomic.ai/map/cb96f41b-a6fd-4fe4-ac40-08e101714483/ae06156c-a572-46e9-8345-ebe18586d02b) |
| **Code**         | [Instructions](https://atlas.nomic.ai/map/902d6ccb-0bbb-4294-83a8-1c7d2dae03c8/ace2e146-e49f-41db-a1f4-25a2c4be2457) | [Tasks](https://atlas.nomic.ai/map/efc38617-9180-490a-8630-43a05b35d22d/2576addf-a133-45d5-89a9-6b067b6652dd) |
| **Misalignment** | [Instructions](https://atlas.nomic.ai/map/5c491035-a26e-4a05-9593-82ffb2c3ab40/2bd98896-894e-4807-9ed8-a203ccb14d5e) | [Tasks](https://atlas.nomic.ai/map/abc357dd-9c04-4913-9541-63e259d7ac1f/825139a4-af66-427c-9d0e-f36b5492ab3f) |

<br>

## Cookbooks (Use Cases)

Practical guides and tutorials for implementing specific functionalities in CAMEL-AI agents and societies.

### 1. Basic Concepts

| Cookbook                                                                                                                              | Description                                                                                                                                                               |
| :------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**                       | A step-by-step guide to building your first agent.                                                                                                                      |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)** | Learn to build a collaborative society of agents.                                                                                                                        |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)**                                         | Best practices for message handling in agents.                                                                                                                            |

### 2. Advanced Features

| Cookbook                                                                                                                             | Description                                                                                                               |
| :------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------ |
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)**                                 | Integrating tools for enhanced functionality.                                                                            |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)**                                | Implementing memory systems in agents.                                                                                      |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)**                                     | Recipes for Retrieval-Augmented Generation.                                                                                |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)**                          | Leveraging knowledge graphs with RAG.                                                                                     |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)**                  | Tools for tracking and managing agents in operations.                                                                     |

### 3. Model Training & Data Generation

| Cookbook                                                                                                                                   | Description                                                                                                                                                                                                 |
| :------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)**              | Learn how to generate data with CAMEL and fine-tune models effectively with Unsloth.                                                                                                                    |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)**                         | Explore how to generate data with real function calls and the Hermes format.                                                                                                                         |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)**                                   | Uncover how to generate CoT data with CAMEL and seamlessly upload it to Huggingface.                                                                                                                     |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)** | Discover how to generate CoT data using CAMEL and SFT Qwen with Unsolth, and seamlessly upload your data and model to Huggingface.                                                                         |

### 4. Multi-Agent Systems & Applications

| Cookbook                                                                                                                               | Description                                                                                                                                                                                    |
| :------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)** | Create role-playing agents for data scraping and reporting.                                                                                                                                  |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)**       | Building a team of agents for collaborative judging.                                                                                                                                       |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** | Build dynamic, temporally-aware knowledge graphs for financial applications using a multi-agent system. It processes financial reports, news articles, and research papers to help traders analyze data, identify relationships, and uncover market insights.  |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)**            | Learn how to build a robust customer service bot for Discord using Agentic RAG.                                                                                                                      |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)**     | Learn how to build a robust customer service bot for Discord using Agentic RAG which supports local deployment.                                                                                     |

### 5. Data Processing

| Cookbook                                                                                                                        | Description                                                                         |
| :------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------- |
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)**                                    | Techniques for agents in video data analysis.                                       |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Explore three methods for extracting and processing data from websites using Firecrawl. |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Learn how to create AI agents that work with your PDFs using Chunkr and Mistral AI.     |

<br>

## Real-World Use Cases

Discover real-world applications of CAMEL’s framework, including:

### 1. Infrastructure Automation

| Use Case                                                                                 | Description                                                                                               |
| :--------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)** | Automate and manage network infrastructure using intelligent agents.                                      |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** |  Use intelligent agents to manage Cloudflare resources dynamically, improving cloud security and performance. |

### 2. Productivity & Business Workflows

| Use Case                                                                                   | Description                                                                                               |
| :----------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)** | Coordinate agents to optimize and manage Airbnb listings and host operations.                             |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Analyze PowerPoint documents and extract insights through multi-agent collaboration.                       |

### 3. Retrieval-Augmented Multi-Agent Chat

| Use Case                                                                                    | Description                                                                                               |
| :------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------- |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)** | Query and understand GitHub codebases through CAMEL agents using RAG, improving developer onboarding.         |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)** | Conversational agents extract and summarize video transcripts, for easier content understanding and repurposing. |

### 4. Video & Document Intelligence

| Use Case                                                                                     | Description                                                                                               |
| :------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | Agents perform OCR on video screenshots to summarize visual content, supporting media monitoring.          |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | CAMEL agents use OCR with Mistral to analyze documents, decreasing manual effort in document analysis. |

### 5. Research & Collaboration

| Use Case                                                                                               | Description                                                                                               |
| :----------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Research agents collaborate on literature reviews, improving efficiency.                             |

<br>

## 🗓️ Events

Stay connected with the CAMEL community through:

*   🎙️ **Community Meetings:** Weekly virtual syncs with the CAMEL team.
*   🏆 **Competitions:** Hackathons, bounty tasks, and coding challenges.
*   🤝 **Volunteer Activities:** Contributions, documentation drives, and mentorship.
*   🌍 **Ambassador Programs:** Represent CAMEL in your university or local tech groups.

>  Want to host or participate in a CAMEL event? Join our [Discord](https://discord.com/invite/CNcNpquyDc) or the [Ambassador Program](https://www.camel-ai.org/community).

<br>

## Contributing to CAMEL

> We welcome contributions! Please read our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) to start contributing.

Help us grow by sharing CAMEL on social media!

<br>

## Community & Contact

*   **[GitHub Issues](https://github.com/camel-ai/camel/issues):** Report bugs, request features, and track development.
*   **[Discord](https://discord.camel-ai.org/):** Get real-time support, chat with the community, and stay updated.
*   **[X (Twitter)](https://x.com/CamelAIOrg):** Follow for updates, AI insights, and key announcements.
*   **[Ambassador Project](https://www.camel-ai.org/community):** Advocate for CAMEL-AI, host events, and contribute content.
*   **WeChat Community:** Scan the QR code below to join our WeChat community.

<div align="center">
    <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
</div>

For other inquiries, please contact: [camel-ai@eigent.ai](mailto:camel-ai@eigent.ai)

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

*   Special thanks to [Nomic AI](https://home.nomic.ai/) for access to their Atlas data exploration tool.
*   We thank Haya Hammoud for designing the initial project logo.
*   We implemented research ideas from the following works; cite the original works if you use these modules:
    *   `TaskCreationAgent`, `TaskPrioritizationAgent`, and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/).
    *   `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094).
    *   `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560).

## License

Licensed under the Apache 2.0 License.

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

*   **Concise Hook:** Starts with a compelling one-sentence hook optimized for SEO.
*   **Clear Structure:** Uses headings for better readability and SEO.
*   **Keyword Optimization:** Naturally integrates relevant keywords like "multi-agent systems," "AI," "framework," "open-source," and "research" throughout.
*   **Bulleted Key Features:** Uses bullet points for clear feature presentation.
*   **Call to Action:** Encourages interaction (Star on GitHub, join Discord, etc.).
*   **Easy Installation:** Highlights the simplicity of installation with `pip`.
*   **Emphasis on Community:** The community aspect is highlighted.
*   **Comprehensive Coverage:** Maintains all of the original information but structures it in a more accessible and SEO-friendly way.
*   **Internal links:** Linking to project's internal pages.
*   **Clearer Callouts:** Added clearer and more inviting calls to action.
*   **Formatted Markdown:**  Improved readability.  Used bolding for emphasis.
*   **Updated Badges:**  Ensured badges are up to date and look good.
*   **Expanded Content:** Included a bit more detail in the "Why use CAMEL" and "What can you build" sections.
*   **Corrected Errors:** Fixed some minor typos or inconsistencies