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

## CAMEL: The Multi-Agent Framework for Groundbreaking AI Research

**CAMEL empowers researchers to explore the scaling laws of AI agents by providing a robust, open-source framework for building and experimenting with multi-agent systems.**  Visit the [original repo](https://github.com/camel-ai/camel) to get started!

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

<p style="line-height: 1.5; text-align: center;"> 🐫 CAMEL is an open-source community dedicated to finding the scaling laws of agents. We believe that studying these agents on a large scale offers valuable insights into their behaviors, capabilities, and potential risks. To facilitate research in this field, we implement and support various types of agents, tasks, prompts, models, and simulated environments.</p>


<br>


Join us ([*Discord*](https://discord.camel-ai.org/) or [*WeChat*](https://ghli.org/camel/wechat.png)) in pushing the boundaries of finding the scaling laws of agents. 

🌟 Star CAMEL on GitHub and be instantly notified of new releases.

</div>

<div align="center">
    <img src="docs/images/star.gif" alt="Star" width="186" height="60">
  </a>
</div>

<br>

## Key Features of CAMEL

*   **Large-Scale Agent Systems:** Simulate up to millions of agents for emergent behavior and scaling law studies.
*   **Dynamic Communication:** Enable real-time interactions and collaboration between agents.
*   **Stateful Memory:** Equip agents with memory for improved decision-making across interactions.
*   **Code-as-Prompt:** Code is written to be clear and readable for both humans and AI agents to interpret.
*   **Diverse Agent Types:** Support a wide range of agent roles, tasks, models, and environments.
*   **Data Generation & Tool Integration:** Automate dataset creation and integrate seamlessly with tools.

<br>

## Why Choose CAMEL for Your Research?

CAMEL is a community-driven research platform with over 100 researchers, chosen for its capabilities in:

*   **Large-Scale Simulation:** Study emergent behaviors in complex environments.
*   **Real-time Agent Collaboration:** Facilitate intricate task solving through seamless interactions.
*   **Contextual Decision-Making:** Enhance agent performance with historical context.
*   **Benchmark Support:** Ensure reliable and reproducible results with standardized benchmarks.
*   **Interdisciplinary Research:** Support diverse experiments with different agent types.
*   **Streamlined Workflows:** Automate data creation and tool integration.

<br>

## What Can You Build with CAMEL?

CAMEL offers a versatile platform for:

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

### Example: ChatAgent with DuckDuckGo Search

1.  **Install web tools:**

    ```bash
    pip install 'camel-ai[web_tools]'
    ```

2.  **Set your OpenAI API key:**

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

3.  **Run this Python code:**

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

Explore our [documentation](https://docs.camel-ai.org) for more in-depth instructions, explore different agent types, and get started on your first project.

We provide a [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing) demo showcasing a conversation between two ChatGPT agents playing roles as a python programmer and a stock trader collaborating on developing a trading bot for stock market.

*   **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**
*   **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)**
*   **[Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)**
*   **[Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)**

### Need Help?

Reach out to us on [CAMEL discord](https://discord.camel-ai.org/) for assistance.

<br>

## Tech Stack

<div align="center">
  <a href="https://docs.camel-ai.org">
    <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="TechStack">
  </a>
</div>

### Key Modules

Explore the core components of CAMEL-AI.

| Module | Description |
| :----- | :---------- |
| **[Agents](https://docs.camel-ai.org/key_modules/agents.html)** | Core agent architectures and behaviors. |
| **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)** | Multi-agent system management. |
| **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)** | Synthetic data creation tools. |
| **[Models](https://docs.camel-ai.org/key_modules/models.html)** | Agent intelligence model options. |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)** | Integrated tools for specialized tasks. |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)** | Agent state management. |
| **[Storage](https://docs.camel-ai.org/key_modules/storages.html)** | Persistent storage solutions. |
| **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)** | Performance evaluation frameworks. |
| **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)** | Code and command interpretation. |
| **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)** | Data ingestion and preprocessing. |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)** | Knowledge retrieval and RAG components. |
| **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)** | Execution environment management. |
| **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Human oversight and intervention. |
---

## Research with CAMEL

Explore our research projects and discover how CAMEL is pushing the boundaries of AI:

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
> We invite you to leverage CAMEL for your impactful research! Join our community and explore the frontiers of Multi-agent Systems by participating in ongoing projects or by testing your ideas. For more information, contact us at [camel-ai@eigent.ai](mailto:camel-ai@eigent.ai).
>
><div align="center">
>    <img src="docs/images/partners.png" alt="Partners">
></div>

<br>

## Synthetic Datasets

### 1. LLM Backends

Explore various LLM options in our [`Models Documentation`](https://docs.camel-ai.org/key_modules/models.html#).

> **Datasets (Hosted on Hugging Face)**

| Dataset        | Chat Format                                                                                         | Instruction Format                                                                                               | Chat Format (translated)                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **AI Society** | [Chat format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_chat.tar.gz) | [Instruction format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_instructions.json) | [Chat format (translated)](https://huggingface.co/datasets/camel-ai/ai_society_translated) |
| **Code**       | [Chat format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_chat.tar.gz)             | [Instruction format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_instructions.json)             | x                                                                                          |
| **Math**       | [Chat format](https://huggingface.co/datasets/camel-ai/math)                                        | x                                                                                                                | x                                                                                          |
| **Physics**    | [Chat format](https://huggingface.co/datasets/camel-ai/physics)                                     | x                                                                                                                | x                                                                                          |
| **Chemistry**  | [Chat format](https://huggingface.co/datasets/camel-ai/chemistry)                                   | x                                                                                                                | x                                                                                          |
| **Biology**    | [Chat format](https://huggingface.co/datasets/camel-ai/biology)                                     | x                                                                                                                | x                                                                                          |

### 2. Visualizations

| Dataset          | Instructions                                                                                                         | Tasks                                                                                                         |
|------------------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **AI Society**   | [Instructions](https://atlas.nomic.ai/map/3a559a06-87d0-4476-a879-962656242452/db961915-b254-48e8-8e5c-917f827b74c6) | [Tasks](https://atlas.nomic.ai/map/cb96f41b-a6fd-4fe4-ac40-08e101714483/ae06156c-a572-46e9-8345-ebe18586d02b) |
| **Code**         | [Instructions](https://atlas.nomic.ai/map/902d6ccb-0bbb-4294-83a8-1c7d2dae03c8/ace2e146-e49f-41db-a1f4-25a2c4be2457) | [Tasks](https://atlas.nomic.ai/map/efc38617-9180-490a-8630-43a05b35d22d/2576addf-a133-45d5-89a9-6b067b6652dd) |
| **Misalignment** | [Instructions](https://atlas.nomic.ai/map/5c491035-a26e-4a05-9593-82ffb2c3ab40/2bd98896-894e-4807-9ed8-a203ccb14d5e) | [Tasks](https://atlas.nomic.ai/map/abc357dd-9c04-4913-9541-63e259d7ac1f/825139a4-af66-427c-9d0e-f36b5492ab3f) |

<br>

## Cookbooks (Use Cases)

Find practical guides for implementing CAMEL-AI functionalities:

### 1. Basic Concepts

| Cookbook                                                                                                                      | Description                                  |
| :---------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------- |
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**            | Build your first agent.                    |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)** | Create a collaborative agent society.     |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)**                                | Best practices for message handling.       |

### 2. Advanced Features

| Cookbook                                                                                                  | Description                             |
| :-------------------------------------------------------------------------------------------------------- | :-------------------------------------- |
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)**       | Integrate tools for enhanced function. |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)**     | Implement memory systems.               |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)**           | Recipes for Retrieval-Augmented Generation. |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)** | Leverage knowledge graphs with RAG.       |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)** | Tracking and managing agents. |

### 3. Model Training & Data Generation

| Cookbook                                                                                                                             | Description                                                                         |
| :----------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- |
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)**           | Generate data and finetune models with Unsloth.                    |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)**                | Generate data using real function calls.            |
| **[CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)**               | Generate CoT data and upload it to Huggingface.  |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)**| Generate CoT data using CAMEL and SFT Qwen with Unsolth and upload to Huggingface. |

### 4. Multi-Agent Systems & Applications

| Cookbook                                                                                                                      | Description                                              |
| :---------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------- |
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)** | Scrape data and generate reports.                          |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)** | Build a judging committee.                          |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** | Dynamic, temporally-aware knowledge graphs.                |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)** | Build a customer service bot for Discord.       |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)** | Build a customer service bot for Discord.       |

### 5. Data Processing

| Cookbook                                                                                                              | Description                                   |
| :-------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------- |
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)**                         | Analyze video data with agents.            |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Extract data from websites.                   |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Agents that work with PDFs.          |

<br>

## Real-World Use Cases

Explore real-world applications of CAMEL across:

### 1 Infrastructure Automation

| Usecase                                                      | Description                                                                                                                              |
| :----------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **[ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)** | Automation of infrastructure tasks in ACI (Application Centric Infrastructure).                                                        |
| **[Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)** | Intelligent agents manage Cloudflare resources for cloud security and performance tuning. |

### 2 Productivity & Business Workflows

| Usecase                                                      | Description                                                                                                                               |
| :----------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| **[Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)** | Agent-driven optimization and management of Airbnb listings and host operations.                                                               |
| **[PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)** | Analyze PowerPoint documents and extract insights through multi-agent collaboration. |

### 3 Retrieval-Augmented Multi-Agent Chat

| Usecase                                                      | Description                                                                                                                                   |
| :----------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)** | Query and understand GitHub codebases through CAMEL agents, accelerating developer onboarding and codebase navigation using RAG. |
| **[Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)** | Conversational agents extract and summarize video transcripts, enabling faster content understanding and repurposing. |

### 4 Video & Document Intelligence

| Usecase                                                      | Description                                                                                                                      |
| :----------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| **[YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)** | Agents perform OCR on video screenshots for content summarization.                                                                   |
| **[Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)** | Use OCR with Mistral to analyze documents. |

### 5 Research & Collaboration

| Usecase                                                      | Description                                                                                                                            |
| :----------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| **[Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)** | Simulate research agents collaborating on literature review for improved efficiency in exploratory analysis. |

<br>

## 🗓️ Events

Join our community events:

*   **Community Meetings:** Weekly virtual syncs.
*   **Competitions:** Hackathons, challenges.
*   **Volunteer Activities:** Contributions and mentorship.
*   **Ambassador Programs:** Represent CAMEL in your local tech groups.

>   Host or participate in a CAMEL event? Join our [Discord](https://discord.com/invite/CNcNpquyDc) or become part of the [Ambassador Program](https://www.camel-ai.org/ambassador).

<br>

## Contributing to CAMEL

>   We welcome contributions! Review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) to get started.
>
>   Share CAMEL on social media to help us grow!

<br>

## Community & Contact

*   **GitHub Issues:** Report bugs, request features: [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** Get support: [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** Follow for updates: [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** Advocate for CAMEL-AI: [Learn more](https://www.camel-ai.org/community)
*   **WeChat Community:** Scan the QR code below.

  <div align="center">
    <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
  </div>
<br>
For more information please contact camel-ai@eigent.ai

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

Please cite the original works if you use the modules: `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.* and `PersonaHub` from *Tao Ge et al.* and `Self-Instruct` from *Yizhong Wang et al.*.

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
```
Key improvements:

*   **SEO Optimization:**  Used keywords like "multi-agent," "AI agents," and "LLM" throughout.
*   **Concise Hook:**  The opening sentence immediately explains what CAMEL is and its purpose.
*   **Clear Headings:**  Used H2/H3 to organize the content logically.
*   **Bulleted Lists:**  Made key features and benefits easy to scan.
*   **Summarized Content:**  Condensed lengthy sections while retaining key information.
*   **Call to Action:** Encourages users to star the repo and join the community.
*   **Emphasis on Benefits:** Highlighted *why* someone should use CAMEL.
*   **Simplified Quick Start:** Makes the instructions even easier to follow.
*   **Comprehensive Structure:** Covers all essential sections from the original README, but re-organized for readability and SEO.
*   **More descriptive titles for use cases.**
*   **Added more specific details to use case descriptions**
*   **Inclusion of Dataset Descriptions.**
*   **Added a note encouraging citation**.