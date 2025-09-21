<!-- Improved and SEO-Optimized README for SkyRL -->

<div align="center">

# SkyRL: The Full-Stack Reinforcement Learning Library for LLMs

[![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/)
[![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL)
[![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI)
[![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI)
[![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN)
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)

</div>

**SkyRL empowers developers to build and train advanced reinforcement learning (RL) agents for Large Language Models (LLMs), with a focus on long-horizon, real-world tasks.**

## Key Features

*   **Modular Design:** SkyRL is built with modularity in mind, allowing for easy customization and integration.
*   **`skyagent` for Agentic Pipelines:** Leverage our agent layer for training LLMs in complex, real-world environments.
*   **`skyrl-train` for High-Performance RL Training:** A modular and performant training framework for all your RL needs.
*   **`skyrl-gym` for LLM-Focused Environments:** A Gymnasium API-compatible environment library featuring math, coding, search, and SQL tasks.

## Getting Started

Dive into the world of RL for LLMs with SkyRL!  Here's how to get started:

*   **Development Guide:** Explore our [Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html) for comprehensive instructions.
*   **Training with `skyrl-train`:** Begin using, modifying, or extending the SkyRL training stack. See the [quickstart docs](https://skyrl.readthedocs.io/en/latest/index.html) for a fast track.
*   **Building Environments with `skyrl-gym`:** Easily integrate your tasks using the simple Gymnasium interface.
*   **Agentic Pipelines with `skyagent`:** Optimize and scale multi-turn tool use LLMs for long-horizon tasks.

## News

*   **[2025/06/26]** 🎉 Released SkyRL-v0.1: A highly-modular, performant RL training framework. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/06/26]** 🎉 Released SkyRL-Gym: A library of RL environments for LLMs implemented with the Gymnasium API. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/05/20]** 🎉 Released SkyRL-SQL: a multi-turn RL training pipeline for Text-to-SQL, along with SkyRL-SQL-7B — a model trained on just 653 samples that outperforms both GPT-4o and o4-mini!
*   **[2025/05/06]** 🎉 Released SkyRL-v0: our open RL training pipeline for multi-turn tool use LLMs, optimized for long-horizon, real-environment tasks like SWE-Bench!

## Links

*   📜 [SkyRL-v0.1 Blog Post](https://novasky-ai.notion.site/skyrl-v01)
*   📜 [SkyRL-SQL Blog Post](https://novasky-ai.notion.site/skyrl-sql)
*   📜 [SkyRL-v0 Blog Post](https://novasky-ai.notion.site/skyrl-v0)

## Acknowledgements

This project is developed at the [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) in collaboration with [**Anyscale**](https://www.anyscale.com/). We gratefully acknowledge the generous compute support from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), and [**AMD**](https://www.amd.com/en.html).

We also acknowledge and appreciate the contributions of the following projects: [veRL](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [OpenReasonerZero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL).

## Citation

If you find SkyRL helpful for your research, please consider citing our work:

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

[Back to the original repository](https://github.com/NovaSky-AI/SkyRL)