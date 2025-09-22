<div align="center">
  <h1>SkyRL: The Full-Stack Reinforcement Learning Library for LLMs</h1>

  [![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/)
  [![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL)
  [![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI)
  [![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI)
  [![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN)
  [![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)
</div>

<hr>

**SkyRL empowers researchers and developers to build and train advanced reinforcement learning agents for Large Language Models (LLMs) across a variety of tasks.**  This comprehensive library offers a modular and flexible framework for developing cutting-edge RL solutions.

## Key Features

*   **Modular Agent Layer (`skyagent`):**  Develop and train intelligent agents for long-horizon, real-world tasks. Includes code for [SkyRL-v0](https://novasky-ai.notion.site/skyrl-v0).
*   **Performant RL Training Framework (`skyrl-train`):** A modular and efficient training framework for RL, enabling rapid experimentation and scaling.
*   **Gymnasium-Compatible Environments (`skyrl-gym`):** Explore a rich set of tool-use tasks, including math, coding, search, and SQL environments, all within the familiar Gymnasium API.
*   **SkyRL-SQL:** Multi-turn RL training pipeline for Text-to-SQL.

## Getting Started

Ready to dive in?

*   Explore the [`skyrl-train`](./skyrl-train) framework to build, modify, and experiment with RL training.  See our [quickstart docs](https://skyrl.readthedocs.io/en/latest/index.html) to get started.
*   Use [`skyrl-gym`](./skyrl-gym) to create your own custom tasks within the Gymnasium interface.
*   Utilize [`skyagent`](./skyagent) to optimize and scale pipelines for multi-turn tool use LLMs on long-horizon, real-environment tasks.
*   Review the [Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html) for comprehensive development instructions.

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

This work is done at [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) in collaboration with [**Anyscale**](https://www.anyscale.com/), with generous compute support from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), and [**AMD**](https://www.amd.com/en.html).

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

**[Back to Top](#skyrl-the-full-stack-reinforcement-learning-library-for-llms)** (link back to the top)

For more details, please visit the original repository: [https://github.com/NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)