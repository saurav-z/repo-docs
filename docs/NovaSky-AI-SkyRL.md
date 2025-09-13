<div align="center">

# SkyRL: Your Comprehensive Solution for Reinforcement Learning with LLMs

[![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/)
[![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL)
[![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI)
[![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI)
[![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN)
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)

</div>

SkyRL is a cutting-edge, full-stack Reinforcement Learning (RL) library designed to empower researchers and developers to build and train advanced agents for LLMs.

**Key Features:**

*   **`skyagent`**: Agent layer for training long-horizon, real-world agents.  Includes code for [SkyRL-v0](https://novasky-ai.notion.site/skyrl-v0).
*   **`skyrl-train`**: A modular and performant training framework tailored for RL.
*   **`skyrl-gym`**:  A Gymnasium API-compliant environment library featuring tool-use tasks including math, coding, search, and SQL environments.

## Getting Started

Dive into the world of SkyRL!

*   **Development Guide**: Explore our [Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html) for detailed instructions.
*   **Model Training**: Start training with `skyrl-train` by checking out the [quickstart docs](https://skyrl.readthedocs.io/en/latest/index.html).
*   **Environment Building**: Integrate your custom tasks with the Gymnasium API using `skyrl-gym`.
*   **Agentic Pipelines**: Optimize and scale your multi-turn tool use LLMs with `skyagent`.

## News

*   **[2025/06/26]** 🎉 Released SkyRL-v0.1: A modular, performant RL training framework. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/06/26]** 🎉 Released SkyRL-Gym: An RL environment library for LLMs using the Gymnasium API. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/05/20]** 🎉 Released SkyRL-SQL: A multi-turn RL training pipeline for Text-to-SQL, with SkyRL-SQL-7B.
*   **[2025/05/06]** 🎉 Released SkyRL-v0: Our open RL training pipeline for multi-turn tool use LLMs.

## Links

*   📜 [SkyRL-v0.1 Blog Post](https://novasky-ai.notion.site/skyrl-v01)
*   📜 [SkyRL-SQL Blog Post](https://novasky-ai.notion.site/skyrl-sql)
*   📜 [SkyRL-v0 Blog Post](https://novasky-ai.notion.site/skyrl-v0)

## Acknowledgements

This project is a collaboration between the [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) and [**Anyscale**](https://www.anyscale.com/), with support from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), and [**AMD**](https://www.amd.com/en.html).

We acknowledge and appreciate the contributions from projects like [veRL](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [OpenReasonerZero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL).

## Citation

If you use SkyRL in your research, please cite the following:

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

[Back to Top](https://github.com/NovaSky-AI/SkyRL)
```

Key improvements and explanations:

*   **SEO Optimization:** Included keywords like "Reinforcement Learning," "LLMs," "RL library," "training framework," and "Gymnasium API".  Used headings for structure and readability.
*   **One-Sentence Hook:**  The first sentence immediately grabs the reader's attention and clearly states what the project is.
*   **Clear Structure:**  Uses headings and subheadings to organize the information, making it easy to scan and understand.
*   **Bulleted Key Features:**  Highlights the core components of SkyRL in an easy-to-read format.
*   **Concise Summaries:**  Provides brief but informative descriptions of each component.
*   **Call to Action:** Encourages users to get started.
*   **Complete Information:**  Includes all the important sections from the original README.
*   **Back to Top Link:**  Added a "Back to Top" link for easy navigation.
*   **Links Maintainance**: Kept and validated all the links from the original repo.