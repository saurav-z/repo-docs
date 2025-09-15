<div align="center">
  <a href="https://github.com/camel-ai/camel">
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

## 🐫 CAMEL: Build and Explore Multi-Agent Systems with Ease

CAMEL is an open-source framework empowering researchers and developers to build, explore, and scale multi-agent systems.  [Explore the CAMEL Framework on GitHub](https://github.com/camel-ai/camel).

<br>

### Key Features

*   **Large-Scale Agent Systems:** Simulate and analyze systems with up to a million agents to understand emergent behaviors and scaling laws.
*   **Dynamic Communication:** Enable real-time interactions and collaboration between agents for complex task solving.
*   **Stateful Memory:** Equip agents with memory to retain context and improve decision-making across extended interactions.
*   **Versatile Benchmarks:** Evaluate agent performance using standardized benchmarks for reliable comparisons.
*   **Diverse Agent Types:** Experiment with a wide variety of agent roles, tasks, models, and environments.
*   **Data Generation & Tool Integration:** Automate large-scale dataset creation and easily integrate various tools.

<br>

### Why Use CAMEL?

CAMEL is a community-driven research initiative, providing a robust platform for advancing multi-agent systems research.  Benefits include:

*   **Scalability:** Supports systems with millions of agents, essential for studying emergent behaviors.
*   **Evolvability:** Facilitates continuous improvement of agents through data generation and environment interaction.
*   **Customizability:** Designed for interdisciplinary experiments, supporting various agent types, roles, and tasks.

<br>

### What Can You Build with CAMEL?

CAMEL facilitates the creation of diverse applications through the following:

*   **Data Generation:** Generate synthetic data for training and evaluation.
    *   [CoT Data Generation](https://github.com/camel-ai/camel/blob/master/camel/datagen/cot_datagen.py)
    *   [Self-Instruct Data Generation](https://github.com/camel-ai/camel/tree/master/camel/datagen/self_instruct)
    *   [Source2Synth Data Generation](https://github.com/camel-ai/camel/tree/master/camel/datagen/source2synth)
    *   [Self-Improving Data Generation](https://github.com/camel-ai/camel/blob/master/camel/datagen/self_improving_cot.py)
*   **Task Automation:** Automate complex tasks with agent collaboration.
    *   [Role Playing](https://github.com/camel-ai/camel/blob/master/camel/societies/role_playing.py)
    *   [Workforce](https://github.com/camel-ai/camel/tree/master/camel/societies/workforce)
    *   [RAG Pipeline](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)
*   **World Simulation:** Simulate realistic environments for agent interaction.
    *   [Oasis Case](https://github.com/camel-ai/oasis)

<br>

### Quick Start

Get started with CAMEL in three easy steps:

1.  **Install:** `pip install camel-ai`
2.  **Install Tools:** `pip install 'camel-ai[web_tools]'`
3.  **Set API Key:** `export OPENAI_API_KEY='YOUR_OPENAI_API_KEY'`

**Example: ChatAgent**

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

For detailed installation and usage, explore our [documentation](https://docs.camel-ai.org).

<br>

### Key Modules

*   **[Agents](https://docs.camel-ai.org/key_modules/agents.html)**: Core agent architectures and behaviors.
*   **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)**: Multi-agent system and collaboration components.
*   **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)**: Synthetic data creation and augmentation tools.
*   **[Models](https://docs.camel-ai.org/key_modules/models.html)**: Model architectures and customization.
*   **[Tools](https://docs.camel-ai.org/key_modules/tools.html)**: Tool integrations.
*   **[Memory](https://docs.camel-ai.org/key_modules/memory.html)**: Agent state management.
*   **[Storage](https://docs.camel-ai.org/key_modules/storages.html)**: Persistent storage.
*   **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)**: Performance evaluation.
*   **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)**: Code and command interpretation.
*   **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)**: Data ingestion and preprocessing.
*   **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)**: Knowledge retrieval and RAG.
*   **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)**: Execution environment.
*   **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)**: Human oversight.

<br>

### Research

CAMEL fuels cutting-edge research in multi-agent systems.

**Explore Our Research Projects:**
*   [OWL](https://github.com/camel-ai/owl)
*   [OASIS](https://oasis.camel-ai.org/)
*   [CRAB](https://crab.camel-ai.org/)
*   [Loong](https://github.com/camel-ai/loong)
*   [Agent Trust](https://agent-trust.camel-ai.org/)
*   [EMOS](https://emos-project.github.io/)

**Join Us:** [Reach out via email](mailto:camel-ai@eigent.ai) to collaborate on ongoing projects.

<div align="center">
    <img src="docs/images/partners.png" alt="Partners">
</div>

<br>

### Synthetic Datasets

*   **AI Society**
    *   [Chat format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_chat.tar.gz)
    *   [Instruction format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_instructions.json)
    *   [Chat format (translated)](https://huggingface.co/datasets/camel-ai/ai_society_translated)
    *   [Instructions](https://atlas.nomic.ai/map/3a559a06-87d0-4476-a879-962656242452/db961915-b254-48e8-8e5c-917f827b74c6)
    *   [Tasks](https://atlas.nomic.ai/map/cb96f41b-a6fd-4fe4-ac40-08e101714483/ae06156c-a572-46e9-8345-ebe18586d02b)
*   **Code**
    *   [Chat format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_chat.tar.gz)
    *   [Instruction format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_instructions.json)
    *   [Instructions](https://atlas.nomic.ai/map/902d6ccb-0bbb-4294-83a8-1c7d2dae03c8/ace2e146-e49f-41db-a1f4-25a2c4be2457)
    *   [Tasks](https://atlas.nomic.ai/map/efc38617-9180-490a-8630-43a05b35d22d/2576addf-a133-45d5-89a9-6b067b6652dd)
*   **Math** [Chat format](https://huggingface.co/datasets/camel-ai/math)
*   **Physics** [Chat format](https://huggingface.co/datasets/camel-ai/physics)
*   **Chemistry** [Chat format](https://huggingface.co/datasets/camel-ai/chemistry)
*   **Biology** [Chat format](https://huggingface.co/datasets/camel-ai/biology)
*   **Misalignment**
    *   [Instructions](https://atlas.nomic.ai/map/5c491035-a26e-4a05-9593-82ffb2c3ab40/2bd98896-894e-4807-9ed8-a203ccb14d5e)
    *   [Tasks](https://atlas.nomic.ai/map/abc357dd-9c04-4913-9541-63e259d7ac1f/825139a4-af66-427c-9d0e-f36b5492ab3f)

<br>

### Cookbooks (Usecases)

*   **Basic Concepts:** Learn the fundamentals of CAMEL.
*   **Advanced Features:** Integrate tools, memory, and RAG.
*   **Model Training & Data Generation:** Generate and fine-tune models.
*   **Multi-Agent Systems & Applications:** Build real-world applications.
*   **Data Processing:** Process data from various sources.

<br>

### Real-World Usecases

Explore practical applications of CAMEL:

*   **Infrastructure Automation:** Manage cloud resources (e.g., [ACI MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/aci_mcp), [Cloudflare](https://github.com/camel-ai/camel/tree/master/examples/usecases/cloudfare_mcp_camel)).
*   **Productivity & Business Workflows:** Optimize operations (e.g., [Airbnb MCP](https://github.com/camel-ai/camel/tree/master/examples/usecases/airbnb_mcp)).
*   **Retrieval-Augmented Multi-Agent Chat:** Create intelligent conversational agents (e.g., [Chat with GitHub](https://github.com/camel-ai/camel/tree/master/examples/usecases/chat_with_github)).
*   **Video & Document Intelligence:** Analyze and summarize content (e.g., [YouTube OCR](https://github.com/camel-ai/camel/tree/master/examples/usecases/youtube_ocr)).
*   **Research & Collaboration:** Simulate research teams (e.g., [Multi-Agent Research Assistant](https://github.com/camel-ai/camel/tree/master/examples/usecases/multi_agent_research_assistant)).

<br>

### 🧱 Built with CAMEL (Real-world Products & Research)

*   **Research Projects**
    *   [ChatDev](https://github.com/OpenBMB/ChatDev/tree/main/camel)
    *   [Paper2Poster](https://github.com/Paper2Poster/Paper2Poster)
*   **Product Projects**
    *   [Eigent](https://www.eigent.ai/)
    *   [EigentBot](https://bot.eigent.ai/)
    *   [Matrix](https://matrix.eigent.ai/)
    *   [AI Geometric](https://www.linkedin.com/posts/aigeometric_ai-interviewpreparation-careerdevelopment-activity-7261428422516555776-MtaK/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAChHluEB9xRwkjiJ6VSAzqM2Y-U4NI2sKGY)
    *   [Log10](https://github.com/log10-io/log10/blob/main/src/log10/agents/camel.py)

<br>

### 🗓️ Events

Join our community events:

*   Community Meetings
*   Competitions
*   Volunteer Activities
*   Ambassador Programs

[Join our Discord](https://discord.com/invite/CNcNpquyDc) or learn more about the [Ambassador Program](https://www.camel-ai.org/ambassador).

<br>

### Contributing to CAMEL

Contribute code by reviewing our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md).

<br>

### Community & Contact

*   **Report Issues:** [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Join Discord:** [Join us](https://discord.camel-ai.org/)
*   **Follow on X:** [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** [Learn more](https://www.camel-ai.org/community)
*   **WeChat:** Scan the QR code below to join our WeChat community.

  <div align="center">
    <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
  </div>

For any other queries, contact camel-ai@eigent.ai

<br>

### Citation

```
@inproceedings{li2023camel,
  title={CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society},
  author={Li, Guohao and Hammoud, Hasan Abed Al Kader and Itani, Hani and Khizbullin, Dmitrii and Ghanem, Bernard},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

<br>

### Acknowledgment

Special thanks to [Nomic AI](https://home.nomic.ai/) and Haya Hammoud.

<br>

### License

Apache 2.0