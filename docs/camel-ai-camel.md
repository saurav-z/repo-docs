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

## CAMEL: Explore the Frontiers of Agent-Based AI

**CAMEL empowers researchers and developers to build and study multi-agent systems, facilitating groundbreaking discoveries in artificial intelligence.** Explore the original repo [here](https://github.com/camel-ai/camel).

### Key Features:

*   ✅ **Large-Scale Agent System:** Simulate up to 1 million agents to explore emergent behaviors and scaling laws.
*   ✅ **Dynamic Communication:** Enable real-time agent interactions for complex task collaboration.
*   ✅ **Stateful Memory:** Equip agents with memory for improved decision-making over time.
*   ✅ **Diverse Agent Types:** Experiment with various agent roles, tasks, models, and environments.
*   ✅ **Data Generation & Tool Integration:** Automate dataset creation and seamlessly integrate with essential tools.
*   ✅ **Community-Driven:** Join a vibrant community of researchers advancing multi-agent system research.

### Key Modules:

*   **[Agents](https://docs.camel-ai.org/key_modules/agents.html)**: Core agent architectures and behaviors.
*   **[Agent Societies](https://docs.camel-ai.org/key_modules/society.html)**: Build and manage multi-agent systems.
*   **[Data Generation](https://docs.camel-ai.org/key_modules/datagen.html)**: Tools for synthetic data creation.
*   **[Models](https://docs.camel-ai.org/key_modules/models.html)**: Model architectures and customization.
*   **[Tools](https://docs.camel-ai.org/key_modules/tools.html)**: Integration for specialized agent tasks.
*   **[Memory](https://docs.camel-ai.org/key_modules/memory.html)**: Agent state management.
*   **[Storage](https://docs.camel-ai.org/key_modules/storages.html)**: Persistent storage solutions.
*   **[Benchmarks](https://github.com/camel-ai/camel/tree/master/camel/benchmarks)**: Performance evaluation.
*   **[Interpreters](https://docs.camel-ai.org/key_modules/interpreters.html)**: Code and command interpretation.
*   **[Data Loaders](https://docs.camel-ai.org/key_modules/loaders.html)**: Data ingestion and preprocessing.
*   **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)**: Knowledge retrieval and RAG components.
*   **[Runtime](https://github.com/camel-ai/camel/tree/master/camel/runtime)**: Execution environment and process management.
*   **[Human-in-the-Loop](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)**: Human oversight and intervention.

### Why Use CAMEL?

CAMEL provides a comprehensive framework for researchers and developers, offering:

*   **Community Support:** Join a community of over 100 researchers.
*   **Reproducibility:**  Use standardized benchmarks for reliable comparisons.
*   **Evolvability:** Continuous evolution of agent systems.
*   **Scalability:** Support for millions of agents.
*   **Code-as-Prompt:** Clear, readable code for both humans and agents.

### Quick Start

Install CAMEL via PyPI:

```bash
pip install camel-ai
```

Example using `ChatAgent`:

1.  Install the tools package:

    ```bash
    pip install 'camel-ai[web_tools]'
    ```

2.  Set your OpenAI API key:

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

3.  Run this Python code:

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

For more detailed instructions, see the [installation section](https://github.com/camel-ai/camel/blob/master/docs/get_started/installation.md).

### What Can You Build with CAMEL?

*   **Data Generation:** (Chain of Thought, Self-Instruct, Source2Synth, Self-Improving)
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

*   **Task Automation:** (Role Playing, Workforce, RAG Pipeline)
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

*   **World Simulation:** (Oasis Case)
    <div align="center">
      <a href="https://github.com/camel-ai/oasis">
        <img src="docs/images/oasis_case.png" alt="Oasis Case">
      </a>
    </div>

### Explore Cookbooks & Use Cases:

*   **Basic Concepts**: Creating Agents & Societies, Message Handling.
*   **Advanced Features**: Tools, Memory, RAG, Graph RAG.
*   **Model Training & Data Generation:** Data Gen with CAMEL, fine-tuning with Unsloth
*   **Multi-Agent Systems & Applications**: Role-Playing Scraper, Hackathon Judge Committee, and more.
*   **Data Processing**: Video analysis, data ingestion from websites and PDFs.

### Synthetic Datasets:

*   **AI Society:** Chat and instruction formats.
*   **Code:** Chat and instruction formats.
*   **Math, Physics, Chemistry, Biology:** Chat formats.

### Real-World Usecases:

*   **Infrastructure Automation:** ACI MCP, Cloudflare MCP CAMEL.
*   **Productivity & Business Workflows:** Airbnb MCP, PPTX Toolkit Usecase.
*   **Retrieval-Augmented Multi-Agent Chat:** Chat with GitHub, Chat with YouTube.
*   **Video & Document Intelligence:** YouTube OCR, Mistral OCR.
*   **Research & Collaboration:** Multi-Agent Research Assistant.

### 🗓️ Events

*   **Community Meetings:** Weekly virtual syncs.
*   **Competitions:** Hackathons and coding challenges.
*   **Volunteer Activities:** Contributions, documentation, and mentorship.
*   **Ambassador Programs:** Represent CAMEL in your community.

### Research:

*   CRAB
    <div align="center">
      <a href="https://crab.camel-ai.org/">
        <img src="docs/images/crab.png" alt="CRAB">
      </a>
    </div>
*   Agent Trust
    <div align="center">
      <a href="https://agent-trust.camel-ai.org/">
        <img src="docs/images/agent_trust.png" alt="Agent Trust">
      </a>
    </div>
*   OASIS
    <div align="center">
      <a href="https://oasis.camel-ai.org/">
        <img src="docs/images/oasis.png" alt="OASIS">
      </a>
    </div>
*   EMOS
    <div align="center">
      <a href="https://emos-project.github.io/">
        <img src="docs/images/emos.png" alt="Emos">
      </a>
    </div>

    **Research with US**
    We warmly invite you to use CAMEL for your impactful research.
    Rigorous research takes time and resources. We are a community-driven research collective with 100+ researchers exploring the frontier research of Multi-agent Systems. Join our ongoing projects or test new ideas with us, [reach out via email](mailto:camel-ai@eigent.ai) for more information.

    <div align="center">
    <img src="docs/images/partners.png" alt="Partners">
    </div>

### Contributing to CAMEL

Review our [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md).

### Community & Contact

*   **GitHub Issues:** [Submit an issue](https://github.com/camel-ai/camel/issues)
*   **Discord:** [Join us](https://discord.camel-ai.org/)
*   **X (Twitter):** [Follow us](https://x.com/CamelAIOrg)
*   **Ambassador Project:** [Learn more](https://www.camel-ai.org/community)
*   **WeChat Community:**  Scan the QR code below to join our WeChat community.

  <div align="center">
    <img src="misc/wechat.jpeg" alt="WeChat QR Code" width="200">
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

### Acknowledgment

Special thanks to [Nomic AI](https://home.nomic.ai/) and Haya Hammoud.

We implemented amazing research ideas from other works for you to build, compare and customize your agents. If you use any of these modules, please kindly cite the original works:
- `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/). [[Example](https://github.com/camel-ai/camel/blob/master/examples/ai_society/babyagi_playing.py)]

- `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094). [[Example](https://github.com/camel-ai/camel/blob/master/examples/personas/personas_generation.py)]

- `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560). [[Example](https://github.com/camel-ai/camel/blob/master/examples/datagen/self_instruct/self_instruct.py)]

### License

Apache 2.0.

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
Key improvements and SEO enhancements:

*   **Clear Heading Structure:** Uses H2 and H3 tags for easy navigation and better organization for both readers and search engines.
*   **SEO-Optimized Title and Introduction:** Includes relevant keywords like "multi-agent systems," "AI research," and "framework" in the title and introductory sentence, and throughout the content.
*   **Bulleted Key Features:**  Uses bullet points to highlight the key features, making them easily scannable. This is important for SEO and user experience.
*   **Concise and Engaging Descriptions:** The descriptions for each feature are short, clear, and benefit-driven.
*   **Keyword Integration:**  Repeatedly uses relevant keywords throughout the document.
*   **Internal Linking:** Links to key modules and cookbooks on the documentation, improving the site's internal linking structure, and SEO.
*   **Call to Action:**  Clear calls to action (e.g., "Join us," "Explore our research projects") encourage engagement.
*   **Consistent Formatting:**  Maintains a consistent style throughout the README.
*   **Complete Summary:**  The revised README includes all the important information from the original while being more concise and better organized.
*   **Image Alt Tags:** Ensures all image tags include appropriate alt text for accessibility and SEO.
*   **Contact Information:** Provides a clear way for users to connect.
*   **Clean Code Blocks:** Uses code blocks for better readability.
*   **Emphasis on Benefits:** Highlights the advantages of using CAMEL.
*   **Community Focus:** Emphasizes the community aspect.