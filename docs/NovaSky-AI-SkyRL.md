<div align="center">

# SkyRL: Train LLMs for Complex Tasks with a Full-Stack Reinforcement Learning Library

[![🌐 NovaSky Website](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/)
[![GitHub](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL)
[![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI)
[![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI)
[![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN)
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)

</div>

**SkyRL empowers researchers and developers to build and train advanced Large Language Models (LLMs) for complex, real-world tasks using a comprehensive, modular Reinforcement Learning (RL) framework.** This library provides the tools you need to train long-horizon agents capable of interacting with real-world environments.

## Key Features

*   **Modular Architecture:** SkyRL is designed with modularity in mind, enabling easy customization and experimentation with different components.

*   **`skyagent`**:  Provides an agent layer for training long-horizon, real-world agents, including code for [SkyRL-v0](https://novasky-ai.notion.site/skyrl-v0).

*   **`skyrl-train`**:  A high-performance, modular training framework for RL, ideal for scaling your projects.

*   **`skyrl-gym`**:  A comprehensive Gymnasium of tool-use environments, including math, coding, search and SQL environments, providing a versatile platform for LLM agent training and evaluation.

## Getting Started

Ready to get started? Jump into the action with these quick links:

*   **Development Guide:**  Explore our [Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html) for a smooth onboarding experience.
*   **Model Training:** Utilize `skyrl-train` for your model training needs. Explore our [quickstart docs](https://skyrl.readthedocs.io/en/latest/index.html) for immediate impact.
*   **Environment Building:** Leverage `skyrl-gym` to build environments within the familiar Gymnasium API.
*   **Agentic Pipelines:** Leverage `skyagent` for optimizing and scaling pipelines for multi-turn tool use LLMs on long-horizon, real-environment tasks.

## News

*   **[2025/06/26]** 🎉 SkyRL-v0.1 Released: A highly-modular, performant RL training framework.  [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/06/26]** 🎉 SkyRL-Gym Released: A library of RL environments for LLMs implemented with the Gymnasium API. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/05/20]** 🎉 SkyRL-SQL Released: A multi-turn RL training pipeline for Text-to-SQL, with SkyRL-SQL-7B exceeding GPT-4o and o4-mini performance!
*   **[2025/05/06]** 🎉 SkyRL-v0 Released: Our open RL training pipeline for multi-turn tool use LLMs, optimized for long-horizon, real-environment tasks.

## Links

*   📜 [SkyRL-v0.1 Blog Post](https://novasky-ai.notion.site/skyrl-v01)
*   📜 [SkyRL-SQL Blog Post](https://novasky-ai.notion.site/skyrl-sql)
*   📜 [SkyRL-v0 Blog Post](https://novasky-ai.notion.site/skyrl-v0)

## Acknowledgements

This work is done at [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) in collaboration with [**Anyscale**](https://www.anyscale.com/), with generous compute support from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), and [**AMD**](https://www.amd.com/en.html).

We adopt many lessons and code from several great projects such as [veRL](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [OpenReasonerZero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). We appreciate each of these teams and their contributions to open-source research!

## Citation

If you use this library, please cite our work using the following BibTex entries:

```bibtex
@misc{cao2025skyrl,
  title     = {SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning},
  author    = {Shiyi Cao and Sumanth Hegde and Dacheng Li and Tyler Griggs and Shu Liu and Eric Tang and Jiayi Pan and Xingyao Wang and Akshay Malik and Graham Neubig and Kourosh Hakhamaneshi and Richard Liaw and Philipp Moritz and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
  year      = {2025},
}
```

```bibtex
@misc{liu2025skyrlsql,
      title={SkyRL-SQL: Matching GPT-4o and o4-mini on Text2SQL with Multi-Turn RL},
      author={Shu Liu and Sumanth Hegde and Shiyi Cao and Alan Zhu and Dacheng Li and Tyler Griggs and Eric Tang and Akshay Malik and Kourosh Hakhamaneshi and Richard Liaw and Philipp Moritz and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
      year={2025},
}
```

```bibtex
@misc{griggs2025skrylv01,
      title={Evolving SkyRL into a Highly-Modular RL Framework},
      author={Tyler Griggs and Sumanth Hegde and Eric Tang and Shu Liu and Shiyi Cao and Dacheng Li and Charlie Ruan and Philipp Moritz and Kourosh Hakhamaneshi and Richard Liaw and Akshay Malik and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
      year={2025},
      note={Notion Blog}
}
```

---

[Back to Top](#skyrl-train-llms-for-complex-tasks-with-a-full-stack-reinforcement-learning-library)
```

Key improvements and SEO considerations:

*   **Clear Title & Hook:**  The title is more descriptive, and the first sentence acts as a clear hook, explaining what SkyRL does.
*   **Keyword Optimization:**  Keywords like "Reinforcement Learning," "LLMs," "Large Language Models," and "RL framework" are used naturally throughout the text.
*   **Structured Headings:**  Uses clear headings and subheadings for readability and SEO.
*   **Bulleted Key Features:**  Provides a concise overview of the core functionalities.
*   **Concise Summaries:**  Replaces lengthy paragraphs with brief summaries of each component.
*   **Call to Action:** The "Getting Started" section encourages user engagement.
*   **Internal Linking:** Links to relevant documentation within the README and to the original repository.
*   **"Back to Top" Link:** Added to the bottom for easy navigation.
*   **Overall Readability:** Enhanced formatting and spacing.
*   **Focus on User Benefits:** Highlights *what* SkyRL allows users to do, rather than just *what it is*.