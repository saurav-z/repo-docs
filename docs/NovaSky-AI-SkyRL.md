<div align="center">

# SkyRL: Revolutionizing RL for LLMs with a Full-Stack, Modular Approach

[![🌐 NovaSky Website](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/)
[![GitHub](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL)
[![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI)
[![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI)
[![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN)
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)

</div>

SkyRL is a cutting-edge, full-stack Reinforcement Learning (RL) library designed specifically for training powerful Language Models (LLMs), offering a modular and efficient approach to complex RL tasks.

## Key Features

*   **`skyagent`**: The agent layer designed for training long-horizon, real-world agents. Includes code for [SkyRL-v0](https://novasky-ai.notion.site/skyrl-v0).
*   **`skyrl-train`**: A modular and performant training framework for RL, providing flexibility and efficiency.
*   **`skyrl-gym`**: A rich Gymnasium of tool-use tasks, including math, coding, search, and SQL environments, all implemented with the Gymnasium API.
*   **Modular Design**: Easily integrate, modify, or build upon existing components.
*   **Open-Source**: Leverage the power of RL for LLMs with an accessible and collaborative framework.

## Getting Started

Ready to dive into the world of RL for LLMs?

*   **Development Guide:** Start with our [Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html) to learn how to develop with SkyRL.
*   **Model Training:** Explore and customize the training stack with [`skyrl-train`](./skyrl-train). Get started quickly with our [quickstart docs](https://skyrl.readthedocs.io/en/latest/index.html).
*   **Environment Building:** Integrate your task with the Gymnasium API using [`skyrl-gym`](./skyrl-gym).
*   **Agentic Pipelines:** Optimize and scale pipelines for multi-turn tool use LLMs on long-horizon, real-environment tasks with [`skyagent`](./skyagent).

## News

*   **[2025/06/26]** 🎉 Released SkyRL-v0.1: A highly-modular, performant RL training framework. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/06/26]** 🎉 Released SkyRL-Gym: A library of RL environments for LLMs implemented with the Gymnasium API. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/05/20]** 🎉 Released SkyRL-SQL: a multi-turn RL training pipeline for Text-to-SQL, along with SkyRL-SQL-7B — a model trained on just 653 samples that outperforms both GPT-4o and o4-mini!
*   **[2025/05/06]** 🎉 Released SkyRL-v0: our open RL training pipeline for multi-turn tool use LLMs, optimized for long-horizon, real-environment tasks like SWE-Bench!

## Links

*   📜 [SkyRL-v0.1 Blog Post](https://novasky-ai.notion.site/skyrl-v01)
*   📜 [SkyRL-SQL Blog Post](https://novasky-ai.notion.site/skyrl-sql)
*   📜 [SkyRL-v0 Blog Post](https://novasky-ai.notion.site/skyrl-v0)

## Acknowledgement

This work is conducted at the [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) in collaboration with [**Anyscale**](https://www.anyscale.com/), and supported by generous compute from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), and [**AMD**](https://www.amd.com/en.html).

We are inspired by projects like [veRL](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [OpenReasonerZero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL).

## Citation

If you find this work helpful, please cite:

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
**[Back to the GitHub Repository](https://github.com/NovaSky-AI/SkyRL)**