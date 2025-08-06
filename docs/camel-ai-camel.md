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

## CAMEL: The Multi-Agent Framework for Exploring Agent Scaling Laws

**CAMEL (Communicative Agents for "Mind" Exploration of Large Language Model Society) is an open-source framework empowering researchers to build, simulate, and study complex multi-agent systems and explore the scaling laws of agents.**  Explore the power of collaborative AI agents and unlock new possibilities for research and development by visiting the [original repo](https://github.com/camel-ai/camel).

**Key Features:**

*   **Large-Scale Agent Systems:** Simulate up to millions of agents.
*   **Dynamic Communication:** Facilitates real-time agent interactions.
*   **Stateful Memory:** Enables agents to retain and use historical context.
*   **Code-as-Prompt:** Leverages code and comments as prompts for agent behavior.
*   **Comprehensive Tool Integration:** Supports various tools for specialized tasks.
*   **Extensive Documentation and Examples:** Provides comprehensive resources for getting started.
*   **Community-Driven:** Benefit from a vibrant and active research community.

**Why Choose CAMEL?**

CAMEL provides a powerful and flexible platform for multi-agent research, offering:

*   **Emergent Behavior Analysis:** Study the emergence of complex behaviors at scale.
*   **Reproducible Research:** Standardized benchmarks ensure reliable comparisons.
*   **Versatile Agent Design:** Support for diverse agent types, roles, models, and environments.
*   **Data Generation and Tool Integration:** Automate dataset creation and streamline workflows.

<br>

## Core Principles

CAMEL is built on several key design principles:

*   **Evolvability:** The framework supports continuous evolution of multi-agent systems through data generation and environment interaction.
*   **Scalability:** Designed to efficiently manage the coordination, communication, and resources of systems with millions of agents.
*   **Statefulness:** Agents maintain stateful memory, enabling multi-step interactions and the tackling of complex tasks.
*   **Code-as-Prompt:** Code serves as prompts for agents, ensuring clarity and interpretability for both humans and agents.

<br>

## Get Started Quickly

Install CAMEL easily via pip:

```bash
pip install camel-ai
```

**Example: Creating a ChatAgent with DuckDuckGo Search**

1.  **Install Web Tools:**

    ```bash
    pip install 'camel-ai[web_tools]'
    ```

2.  **Set Your OpenAI API Key:**

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

3.  **Run this Python Code:**

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

Find detailed instructions and configuration options in the [installation documentation](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).

Explore the [CAMEL Tech Stack](https://docs.camel-ai.org) and [Cookbooks](https://docs.camel-ai.org) to build powerful multi-agent systems.

Explore different agent types, roles, and applications through example cookbooks:

*   **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**
*   **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)**
*   **[Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)**
*   **[Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)**

<br>

## Tech Stack & Modules

**CAMEL offers a comprehensive suite of modules for building and managing multi-agent systems:**

<div align="center">
  <a href="https://docs.camel-ai.org">
    <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="TechStack">
  </a>
</div>

### Key Modules

*   **[Agents](https://docs.camel-ai.org/key_modules/agents.html):** Core agent architectures and behaviors.
*   **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html):** Components for multi-agent system management.
*   **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html):** Tools for creating synthetic data.
*   **[Models](https://docs.camel-ai.org/key_modules/models.html):** Model architectures and customization.
*   **[Tools](https://docs.camel-ai.org/key_modules/tools.html):** Integrations for agent tasks.
*   **[Memory](https://docs.camel-ai.org/key_modules/memory.html):** State management.
*   **[Storage](https://docs.camel-ai.org/key_modules/storages.html):** Data and state persistence.
*   **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks):** Performance evaluation frameworks.
*   **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html):** Code/command interpretation.
*   **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html):** Data ingestion/preprocessing.
*   **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html):** Knowledge retrieval/RAG.
*   **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime):** Execution environment management.
*   **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html):** Human oversight components.

<br>

## Research

**CAMEL enables cutting-edge research on multi-agent systems.**

Explore the team's research projects:

*   [CRAB](https://crab.camel-ai.org/)
*   [Agent Trust](https://agent-trust.camel-ai.org/)
*   [OASIS](https://oasis.camel-ai.org/)
*   [Emos](https://emos-project.github.io/)

>   **Research with Us:** We welcome contributions to our research. [Contact us](mailto:camel-ai@eigent.ai) to join our community.
>   <div align="center">
>       <img src="docs/images/partners.png" alt="Partners">
>   </div>

<br>

## Synthetic Datasets

CAMEL provides various synthetic datasets hosted on Hugging Face for various applications:

*   **AI Society:** [Chat](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_chat.tar.gz) | [Instructions](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_instructions.json) | [Translated Chat](https://huggingface.co/datasets/camel-ai/ai_society_translated)
*   **Code:** [Chat](https://huggingface.co/datasets/camel-ai/code/blob/main/code_chat.tar.gz) | [Instructions](https://huggingface.co/datasets/camel-ai/code/blob/main/code_instructions.json)
*   **Math:** [Chat](https://huggingface.co/datasets/camel-ai/math)
*   **Physics:** [Chat](https://huggingface.co/datasets/camel-ai/physics)
*   **Chemistry:** [Chat](https://huggingface.co/datasets/camel-ai/chemistry)
*   **Biology:** [Chat](https://huggingface.co/datasets/camel-ai/biology)

Visualizations of instructions and tasks are available for each dataset.

<br>

## Cookbooks (Usecases)

**CAMEL provides practical guides to help you implement key functionalities:**

### 1. Basic Concepts

*   [Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)
*   [Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)
*   [Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)

### 2. Advanced Features

*   [Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)
*   [Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)
*   [RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)
*   [Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)
*   [Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)

### 3. Model Training & Data Generation

*   [Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)
*   [Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)
*   [CoT Data Generation and Upload Data to Huggingface](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)
*   [CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)

### 4. Multi-Agent Systems & Applications

*   [Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)
*   [Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)
*   [Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)
*   [Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)
*   [Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)

### 5. Data Processing

*   [Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)
*   [3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)
*   [Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)

<br>

## Real-World Usecases

**Explore real-world applications of the CAMEL framework:**

### 1 Infrastructure Automation

*   [ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp)
*   [Cloudflare MCP CAMEL](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)

### 2 Productivity & Business Workflows

*   [Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)
*   [PPTX Toolkit Usecase](https://github.com/camel-ai/camel/tree/master/examples/usecases/pptx_toolkit_usecase)

### 3 Retrieval-Augmented Multi-Agent Chat

*   [Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)
*   [Chat with YouTube](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_youtube)

### 4 Video & Document Intelligence

*   [YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)
*   [Mistral OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/mistral_OCR)

### 5 Research & Collaboration

*   [Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)

<br>

## Events

**Stay connected and engaged with the CAMEL community:**

*   🎙️ **Community Meetings:** Weekly virtual syncs
*   🏆 **Competitions:** Hackathons and coding challenges
*   🤝 **Volunteer Activities:** Contributions and mentorship
*   🌍 **Ambassador Programs:** Represent CAMEL at your university or in your local tech groups

>   Join our [Discord](https://discord.com/invite/CNcNpquyDc) to host or participate in an event, or learn more about the [Ambassador Program](https://www.camel-ai.org/ambassador).

<br>

## Contributing

>   We welcome contributions!  Review the [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) to get started.

Help us grow the community by sharing CAMEL!

<br>

## Community & Contact

*   **GitHub Issues:** Report bugs and request features: [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** Get support and chat: [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** Follow for updates: [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** Advocate for CAMEL-AI: [Learn more](https://www.camel-ai.org/community)
*   **WeChat Community:** Scan the QR code.

  <div align="center">
    <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
  </div>

*   **Email:** For more information: camel-ai@eigent.ai

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

## Acknowledgments

Special thanks to [Nomic AI](https://home.nomic.ai/) and Haya Hammoud.

Please cite the original works if you use the modules:

*   `TaskCreationAgent`, `TaskPrioritizationAgent`, and `BabyAGI` from *Nakajima et al.*
*   `PersonaHub` from *Tao Ge et al.*
*   `Self-Instruct` from *Yizhong Wang et al.*

<br>

## License

Licensed under the Apache 2.0 License.