<div align="center">

# SkyRL: The Premier Full-Stack RL Library for LLMs

SkyRL empowers researchers and developers to build cutting-edge reinforcement learning (RL) agents for Large Language Models (LLMs) and real-world applications.

[![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/) [![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL) [![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI) [![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI) [![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN) [![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)

</div>

## Key Features

*   **Modular Design:** SkyRL provides a flexible and scalable architecture for RL development, allowing you to easily customize and integrate components.
*   **`skyagent`**:  Train long-horizon, real-world agents with our dedicated agent layer.
*   **`skyrl-train`**:  A high-performance, modular training framework tailored for RL tasks.
*   **`skyrl-gym`**:  A comprehensive suite of tool-use tasks, including environments for math, coding, search, and SQL, all compatible with the Gymnasium API.
*   **Cutting-Edge Research:**  Based on innovative research, including work in multi-turn Text-to-SQL RL.
*   **Open-Source:**  Leverage an open-source library, fostering collaboration and community-driven development.

## Overview

SkyRL is a comprehensive RL library designed to accelerate the development of intelligent agents, particularly for LLMs. It provides a full-stack solution, encompassing agent architectures, training frameworks, and a rich set of environments.

## Getting Started

Explore SkyRL's capabilities with these resources:

*   **Development Guide:**  Get started with our detailed [Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html).
*   **Training with `skyrl-train`:**  Use, modify, or build upon the SkyRL training stack.  See our [quickstart docs](https://skyrl.readthedocs.io/en/latest/index.html) for rapid onboarding.
*   **Environment Building with `skyrl-gym`:** Integrate your tasks using the Gymnasium API.
*   **Agentic Pipelines with `skyagent`:** Optimize and scale pipelines for multi-turn tool use LLMs on long-horizon, real-environment tasks.

## News

*   **[2025/06/26]** 🎉 Released SkyRL-v0.1: Highly-modular, performant RL training framework. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/06/26]** 🎉 Released SkyRL-Gym: RL environments for LLMs implemented with the Gymnasium API. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/05/20]** 🎉 Released SkyRL-SQL: Multi-turn RL pipeline for Text-to-SQL, and SkyRL-SQL-7B, a model trained on 653 samples that outperforms GPT-4o and o4-mini!
*   **[2025/05/06]** 🎉 Released SkyRL-v0: Open RL training pipeline for multi-turn tool use LLMs, optimized for long-horizon, real-environment tasks like SWE-Bench.

## Links

*   📜 [SkyRL-v0.1 Blog Post](https://novasky-ai.notion.site/skyrl-v01)
*   📜 [SkyRL-SQL Blog Post](https://novasky-ai.notion.site/skyrl-sql)
*   📜 [SkyRL-v0 Blog Post](https://novasky-ai.notion.site/skyrl-v0)

## Acknowledgement

This work is done at [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) in collaboration with [**Anyscale**](https://www.anyscale.com/), with generous compute support from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), and [**AMD**](https://www.amd.com/en.html).

We adopt many lessons and code from several great projects such as [veRL](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [OpenReasonerZero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). We appreciate each of these teams and their contributions to open-source research!

## Citation

If you find the work in this repository helpful, please consider citing:

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

[Back to the Top](https://github.com/NovaSky-AI/SkyRL)