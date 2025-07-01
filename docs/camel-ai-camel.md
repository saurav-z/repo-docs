<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="docs/images/banner.png" alt="CAMEL AI Banner">
  </a>
</div>

</br>

<div align="center">
  <a href="https://github.com/camel-ai/camel" target="_blank">
    <img src="https://img.shields.io/github/stars/camel-ai/camel?style=social&label=Star%20on%20GitHub&logo=github" alt="GitHub stars">
  </a>
  <a href="https://discord.camel-ai.org/" target="_blank">
    <img src="https://img.shields.io/discord/1082486657678311454?logo=discord&label=Discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb" alt="Discord">
  </a>
</div>

<hr>

## CAMEL: Unleash the Power of Multi-Agent Systems for Cutting-Edge Research

**CAMEL is an open-source framework empowering researchers to explore the scaling laws of agents and build advanced multi-agent systems.** Discover the [original repository](https://github.com/camel-ai/camel) for more information.

### Key Features:

*   ✅ **Large-Scale Agent Systems:** Simulate up to millions of agents to study emergent behaviors and scaling laws.
*   ✅ **Dynamic Communication:** Facilitate real-time interactions for seamless collaboration on complex tasks.
*   ✅ **Stateful Memory:** Equip agents with memory to retain context and improve decision-making over time.
*   ✅ **Code-as-Prompt Design:** Leverage clear and readable code for human and agent interpretation.
*   ✅ **Versatile Agent Types:** Support a wide range of agent roles, tasks, models, and environments.
*   ✅ **Data Generation & Tool Integration:** Automate dataset creation and streamline research workflows.

### Why Choose CAMEL?

CAMEL is a community-driven research initiative with 100+ researchers advancing the frontier of Multi-Agent Systems research.

### Build with CAMEL

*   **Data Generation:** Create large-scale, structured datasets for agent training.

    <div align="center">
        <img src="docs/images/cot.png" alt="CoT Data Generation" width="200">
        <img src="docs/images/self_instruct.png" alt="Self-Instruct Data Generation" width="200">
        <img src="docs/images/source2synth.png" alt="Source2Synth Data Generation" width="200">
        <img src="docs/images/self_improving.png" alt="Self-Improving Data Generation" width="200">
    </div>

*   **Task Automation:** Automate complex tasks through agent collaboration.

    <div align="center">
        <img src="docs/images/role_playing.png" alt="Role Playing" width="200">
        <img src="docs/images/workforce.png" alt="Workforce" width="200">
        <img src="docs/images/rag_pipeline.png" alt="RAG Pipeline" width="200">
    </div>

*   **World Simulation:** Simulate and explore various real-world scenarios.

    <div align="center">
        <img src="docs/images/oasis_case.png" alt="Oasis Case" width="200">
    </div>

### Quick Start

Easily install CAMEL via PyPI:

```bash
pip install camel-ai
```

Explore the capabilities of CAMEL with a basic ChatAgent example:

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

    response_2 = agent.step("What is the Github link to CAMEL framework?")
    print(response_2.msgs[0].content)
    ```

For detailed setup and configurations, consult the [installation section](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).

Access comprehensive guides and examples:
*   **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**
*   **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)**
*   **[Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)**
*   **[Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)**

### Explore the Tech Stack:
<div align="center">
  <img src="https://camel-ai.github.io/camel_asset/graphics/techstack.png" alt="TechStack">
</div>

**Key Modules**
Core components and utilities to build, operate, and enhance CAMEL-AI agents and societies.

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

### Research

Explore our ongoing research projects:

<div align="center">
    <img src="docs/images/crab.png" alt="CRAB" width="200">
    <img src="docs/images/agent_trust.png" alt="Agent Trust" width="200">
    <img src="docs/images/oasis.png" alt="OASIS" width="200">
    <img src="docs/images/emos.png" alt="Emos" width="200">
</div>

### Synthetic Datasets

Leverage pre-built datasets for training and experimentation.  Explore options from Hugging Face.

### Cookbooks
Practical guides and tutorials for implementing specific functionalities in CAMEL-AI agents and societies.

### Contributing

Join the CAMEL community! Review the [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md).

### Community & Contact

*   **GitHub Issues:** Report bugs and request features. [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** Get real-time support. [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** Stay updated. [Follow us](https://x.com/CamelAIOrg)
*   **WeChat:** Join our community.

    <div align="center">
      <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="150">
    </div>

### Citation

```
@inproceedings{li2023camel,
  title={CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society},
  author={Li, Guohao and Hammoud, Hasan Abed Al Kader and Itani, Hani and Khizbullin, Dmitrii and Ghanem, Bernard},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

### License

Licensed under the Apache 2.0 License.