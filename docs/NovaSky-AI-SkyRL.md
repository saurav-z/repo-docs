<div align="center">

# SkyRL: The Full-Stack Reinforcement Learning Library for LLMs

<p>
    <a href="#overview" style="text-decoration: none; font-weight: bold;">Overview</a> •
    <a href="#key-features" style="text-decoration: none; font-weight: bold;">Key Features</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">Getting Started</a> •
    <a href="#news" style="text-decoration: none; font-weight: bold;">News</a> •
    <a href="#links" style="text-decoration: none; font-weight: bold;">Links</a> •
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">Acknowledgement</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a>
</p>

[![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/)
[![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL)
[![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI)
[![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI)
[![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN)
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)

</div>

## Overview

**SkyRL is a comprehensive, modular, and high-performance reinforcement learning library specifically designed for training and deploying LLMs for real-world tasks.**  This library provides the building blocks needed to create advanced RL agents, environments, and training pipelines.

## Key Features

*   **`skyagent`:** Agent layer optimized for long-horizon, real-world agent training, including the code behind [SkyRL-v0](https://novasky-ai.notion.site/skyrl-v0).
*   **`skyrl-train`:** A modular and performant training framework for RL, enabling flexible experimentation and efficient model training.
*   **`skyrl-gym`:** A Gymnasium-compatible environment library with tool-use tasks, including math, coding, search, and SQL environments.

## Getting Started

Dive into the world of RL with SkyRL! Here’s how to get started:

*   **Development:** Explore our [Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html) for in-depth instructions.
*   **Model Training:** Utilize `skyrl-train` to begin training, modifying, or building upon the SkyRL training stack.  Consult our [quickstart docs](https://skyrl.readthedocs.io/en/latest/index.html) for rapid onboarding.
*   **Environment Building:** Integrate your own tasks using the familiar Gymnasium API within `skyrl-gym`.
*   **Agentic Pipelines:** Explore `skyagent` to optimize and scale your pipelines for multi-turn tool use LLMs in complex, real-world scenarios.

## News

*   **[2025/06/26]** 🎉 Released SkyRL-v0.1: Highly-modular, performant RL training framework. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/06/26]** 🎉 Released SkyRL-Gym: RL environments for LLMs implemented with the Gymnasium API. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
*   **[2025/05/20]** 🎉 Released SkyRL-SQL: Multi-turn RL training pipeline for Text-to-SQL, with SkyRL-SQL-7B, which outperforms GPT-4o and o4-mini.
*   **[2025/05/06]** 🎉 Released SkyRL-v0: Open RL training pipeline for multi-turn tool use LLMs, optimized for long-horizon, real-environment tasks.

## Links

*   📜 [SkyRL-v0.1 Blog Post](https://novasky-ai.notion.site/skyrl-v01)
*   📜 [SkyRL-SQL Blog Post](https://novasky-ai.notion.site/skyrl-sql)
*   📜 [SkyRL-v0 Blog Post](https://novasky-ai.notion.site/skyrl-v0)

## Acknowledgement

This work is done at [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) in collaboration with [**Anyscale**](https://www.anyscale.com/), with generous compute support from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), and [**AMD**](https://www.amd.com/en.html).

We adopt many lessons and code from several great projects such as [veRL](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [OpenReasonerZero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). We appreciate each of these teams and their contributions to open-source research!

## Citation

If you find our work helpful, please consider citing:

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

*   **Concise Hook:** The first sentence is designed to immediately grab the reader's attention and highlight the core value proposition: A reinforcement learning library specifically for LLMs.
*   **Clear Headings:** Uses HTML-style headings and internal links for easy navigation, which improves SEO and readability.
*   **Bulleted Key Features:** Clearly outlines the core components of SkyRL using bullet points, making the information easily digestible.
*   **Keywords:**  Incorporates relevant keywords like "Reinforcement Learning," "LLMs," "Agents," "Environments," "Training," "Modular," and component names.
*   **Action-Oriented Language:**  Uses phrases like "Explore," "Dive into," "Utilize," etc., to encourage engagement.
*   **Internal Linking:** The "Back to Top" link provides convenient navigation.
*   **Summarized Content:** Removed redundant text and consolidated information for clarity.
*   **SEO-Friendly Formatting:**  Uses headings and bullet points to structure the information, which is beneficial for search engine optimization.
*   **Cleaned Up Visuals:** Made the formatting cleaner and easier to read, reducing any clutter.
*   **Direct Links:** Included the direct link at the end.