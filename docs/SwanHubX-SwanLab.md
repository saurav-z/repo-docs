<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: The Open-Source Deep Learning Training Tracker & Visualizer

**SwanLab** empowers researchers to streamline their deep learning experiments with intuitive tracking, visualization, and collaboration tools. Easily integrate with 30+ popular frameworks and run experiments locally or in the cloud.  [Explore SwanLab on GitHub](https://github.com/SwanHubX/SwanLab)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)

---

![](readme_files/swanlab-overview.png)

[中文](README_CN.md) / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

---

## Key Features

*   **Experiment Tracking & Visualization:**
    *   Track key metrics, hyperparameters, and model artifacts with a simple API.
    *   Visualize training progress with interactive charts (line charts, media, custom charts).
    *   Supports various data types: scalar metrics, images, audio, text, video, 3D point clouds, molecule data.
    *   Visualize LLM generated content.
    *   View hardware metrics (CPU, GPU, NPU, memory, disk, network).
*   **Framework Integrations:**
    *   Seamless integration with 30+ popular deep learning frameworks including PyTorch, TensorFlow, Hugging Face Transformers, and more.
*   **Hardware Monitoring:**
    *   Real-time monitoring of system resources, including CPU, GPU, memory, disk, and network usage.
*   **Experiment Management & Collaboration:**
    *   Centralized dashboard for managing projects and experiments.
    *   Compare and analyze results across different experiments.
    *   Collaborate with your team through online sharing and discussion.
*   **Flexibility & Extensibility:**
    *   Supports cloud and offline use.
    *   Plugin support for extending functionality (notifications, data writers, etc.).
    *   Self-hosted option for complete control.
    *   Sync with Tensorboard and Weights & Biases

---

## Getting Started

1.  **Install:**

    ```bash
    pip install swanlab
    ```
2.  **Login:**

    ```bash
    swanlab login
    ```
    (Follow the prompts to get your API key from [swanlab.cn](https://swanlab.cn))
3.  **Integrate with your code:**

    ```python
    import swanlab

    # Initialize a new swanlab experiment
    swanlab.init(
        project="my-first-ml",
        config={'learning-rate': 0.003},
    )

    # Log metrics
    for i in range(10):
        swanlab.log({"loss": i, "acc": i})
    ```

---

## Further Information

*   [Online Demo](https://swanlab.cn)
*   [Documentation](https://docs.swanlab.cn)
*   [Self-Hosting](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)
*   [Examples](https://docs.swanlab.cn/examples)
*   [API Reference](https://docs.swanlab.cn/api/)
*   [Contributing](CONTRIBUTING.md)