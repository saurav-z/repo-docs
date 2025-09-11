<div align="center">

# SkyRL: The Full-Stack RL Library for Training LLMs for Real-World Tasks

[![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/)
[![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL)
[![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI)
[![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI)
[![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN)
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)

</div>

SkyRL empowers researchers and developers to build and train advanced reinforcement learning (RL) agents for Large Language Models (LLMs) capable of handling real-world, long-horizon tasks.

## Key Features

*   **Modular Architecture:** SkyRL is designed with a modular approach, allowing for flexibility and customization in your RL projects.
*   **`skyagent`**:  Our agent layer for training long-horizon, real-world agents, including code for [SkyRL-v0](https://novasky-ai.notion.site/skyrl-v0).
*   **`skyrl-train`**: A modular, performant training framework for RL, offering a robust foundation for your RL training needs.
*   **`skyrl-gym`**:  A Gymnasium of tool-use environments, including math, coding, search, and SQL environments built on the Gymnasium API.
*   **Focus on Long-Horizon Tasks:** Optimized for training agents to solve complex, multi-turn tasks in real-world environments.
*   **Performance and Scalability:** Designed for efficient training and scaling of RL models.

## Getting Started

Ready to dive in? Check out these resources:

*   **[Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html):** Learn how to develop with SkyRL.
*   **`skyrl-train`:** Start using, modifying, or building on top of the SkyRL training stack. See our [quickstart docs](https://skyrl.readthedocs.io/en/latest/index.html).
*   **`skyrl-gym`:** Integrate your own tasks using the familiar Gymnasium API.
*   **`skyagent`:** Optimize and scale pipelines for multi-turn tool use LLMs on long-horizon, real-environment tasks.

## News

*   **June 26, 2025:** Released SkyRL-v0.1: A highly-modular, performant RL training framework. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **June 26, 2025:** Released SkyRL-Gym: A library of RL environments for LLMs implemented with the Gymnasium API. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **May 20, 2025:** Released SkyRL-SQL: a multi-turn RL training pipeline for Text-to-SQL, with SkyRL-SQL-7B — a model trained on just 653 samples that outperforms both GPT-4o and o4-mini!
*   **May 06, 2025:** Released SkyRL-v0: our open RL training pipeline for multi-turn tool use LLMs, optimized for long-horizon, real-environment tasks like SWE-Bench!

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

[**Back to Top**](https://github.com/NovaSky-AI/SkyRL)
```
Key improvements and explanations:

*   **SEO-Optimized Title and Description:**  The title and first sentence now include relevant keywords like "Full-Stack RL," "LLMs," and "Real-World Tasks," increasing search engine visibility.
*   **Clear Headings:**  Uses headings for better readability and organization.
*   **Bulleted Key Features:**  Highlights the most important features with bullet points for easy scanning.
*   **Concise Summary:**  Provides a quick overview of the library's purpose and benefits.
*   **Call to Action:** Encourages users to explore resources.
*   **Internal Links:** Added internal links for quicker navigation.
*   **Cleaned Up Presentation:** Used markdown formatting for consistent look.
*   **Back to Top Link:** Added a link back to the top of the README for easy navigation.
*   **Keyword Density:** The use of relevant terms ("RL," "LLM," "training," "agents," etc.) is balanced throughout the document to improve search ranking without being excessive.