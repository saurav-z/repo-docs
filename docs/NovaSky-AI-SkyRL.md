<div align="center">

# SkyRL: Unleash the Power of Reinforcement Learning for LLMs

[![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/)
[![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL)
[![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI)
[![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI)
[![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN)
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)

</div>

**SkyRL is a full-stack, modular Reinforcement Learning (RL) library designed to empower developers to train and deploy intelligent agents for Large Language Models (LLMs).**  ([Original Repository](https://github.com/NovaSky-AI/SkyRL))

## Key Features:

*   **`skyrl-train`**: A modular and performant training framework for RL, built for efficiency and scalability.
*   **`skyrl-gym`**: A comprehensive suite of RL environments built on the Gymnasium API, including environments for math, coding, search, and SQL tasks.
*   **`skyagent`**: Agent layer for training long-horizon, real-world agents, optimized for multi-turn tool use LLMs.

## Getting Started

Explore SkyRL and start building your RL agents!

*   **Development:**  Check out our [Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html) in our docs.
*   **Model Training:** Use the `skyrl-train` framework. See our [quickstart docs](https://skyrl.readthedocs.io/en/latest/index.html) for initial setup and use.
*   **Environment Building:** Integrate your tasks using the `skyrl-gym` environment.
*   **Agentic Pipelines:** Utilize `skyagent` for pipeline optimization and scaling for multi-turn tool use LLMs.

## News

*   **[2025/06/26]** 🎉 Released SkyRL-v0.1: Highly-modular, performant RL training framework. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/06/26]** 🎉 Released SkyRL-Gym: A library of RL environments for LLMs using the Gymnasium API. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/05/20]** 🎉 Released SkyRL-SQL: Multi-turn RL training pipeline for Text-to-SQL, including SkyRL-SQL-7B, which outperforms GPT-4o and o4-mini with only 653 samples!
*   **[2025/05/06]** 🎉 Released SkyRL-v0: Open RL training pipeline for multi-turn tool use LLMs, optimized for long-horizon tasks like SWE-Bench.

## Links

*   📜 [SkyRL-v0.1 Blog Post](https://novasky-ai.notion.site/skyrl-v01)
*   📜 [SkyRL-SQL Blog Post](https://novasky-ai.notion.site/skyrl-sql)
*   📜 [SkyRL-v0 Blog Post](https://novasky-ai.notion.site/skyrl-v0)

## Acknowledgement

This project is a collaboration between the [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) and [**Anyscale**](https://www.anyscale.com/), with generous support from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), and [**AMD**](https://www.amd.com/en.html).

We would also like to acknowledge the contributions from the following projects: [veRL](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [OpenReasonerZero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL).

## Citation

If you use this work, please cite it as:

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