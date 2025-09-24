<div align="center">
  <a href="https://github.com/NovaSky-AI/SkyRL">
    <img src="https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white" alt="GitHub"/>
  </a>
  <a href="https://novasky-ai.github.io/">
    <img src="https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge" alt="Website"/>
  </a>
  <a href="https://x.com/NovaSkyAI">
    <img src="https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white" alt="Twitter"/>
  </a>
  <a href="https://huggingface.co/NovaSky-AI">
    <img src="https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor" alt="Hugging Face"/>
  </a>
  <a href="https://discord.gg/cJF2JUaaAN">
    <img src="https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"/>
  </a>
  <a href="https://skyrl.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"/>
  </a>
</div>

# SkyRL: Revolutionizing Reinforcement Learning for LLMs

SkyRL is a full-stack, modular reinforcement learning (RL) library designed to train and deploy advanced, real-world agents using Large Language Models (LLMs).

**[Explore the SkyRL Repository on GitHub](https://github.com/NovaSky-AI/SkyRL)**

## Key Features

*   **Modular Training Framework:** `skyrl-train` offers a flexible and performant training environment for RL, designed for experimentation and rapid development.
*   **Gymnasium Environments:**  `skyrl-gym` provides a diverse set of RL environments built with the Gymnasium API, including tasks for math, coding, search, and SQL, enabling you to train your LLM-based agents on a wide range of challenges.
*   **Agent Layer:** `skyagent` supports multi-turn tool use LLMs and long-horizon tasks, empowering you to build cutting-edge agentic pipelines.
*   **Open Source and Community Driven:**  Benefit from a collaborative community, open-source development, and a commitment to advancing RL for LLMs.

## Getting Started

*   **Development:** Explore the [Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html) for instructions on contributing to SkyRL.
*   **Model Training:** Begin your journey with `skyrl-train` by referring to the [quickstart documentation](https://skyrl.readthedocs.io/en/latest/index.html).
*   **Environment Building:** Integrate your tasks using the Gymnasium API with `skyrl-gym`.
*   **Agentic Pipelines:** Use `skyagent` for multi-turn tool use LLMs and long-horizon tasks.

## News and Updates

*   **[2025/06/26]** 🎉 SkyRL-v0.1: A highly-modular, performant RL training framework. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/06/26]** 🎉 SkyRL-Gym: A library of RL environments for LLMs implemented with the Gymnasium API. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/05/20]** 🎉 SkyRL-SQL: A multi-turn RL training pipeline for Text-to-SQL, achieving state-of-the-art results.
*   **[2025/05/06]** 🎉 SkyRL-v0: Open RL training pipeline for multi-turn tool use LLMs optimized for long-horizon, real-environment tasks.

## Links

*   📜 [SkyRL-v0.1 Blog Post](https://novasky-ai.notion.site/skyrl-v01)
*   📜 [SkyRL-SQL Blog Post](https://novasky-ai.notion.site/skyrl-sql)
*   📜 [SkyRL-v0 Blog Post](https://novasky-ai.notion.site/skyrl-v0)

## Acknowledgements

This project is the result of collaboration with the Berkeley Sky Computing Lab in conjunction with Anyscale, with support from Anyscale, Databricks, NVIDIA, Lambda Labs, and AMD.

We would like to thank veRL, OpenRLHF, Search-R1, OpenReasonerZero, and NeMo-RL for their contributions to open-source research.

## Citation

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