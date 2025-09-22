<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## 🚀 SwanLab: Your Open-Source AI Training Tracker and Visualizer

**SwanLab is a modern, open-source tool designed to revolutionize your deep learning workflow, offering seamless integration with 30+ popular frameworks, cloud/offline support, and powerful visualization capabilities.**  Track experiments, visualize metrics, and collaborate on your AI projects with ease.

[🔥 SwanLab Online](https://swanlab.cn) | [📃 Documentation](https://docs.swanlab.cn) | [🙋‍♀️ Report Issues](https://github.com/swanhubx/swanlab/issues) | [💡 Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) | [📜 Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) |  <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> [基线社区](https://swanlab.cn/benchmarks)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

<br/>

![](readme_files/swanlab-overview.png)

中文 / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [微信群](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **📊 Comprehensive Experiment Tracking:** Easily track key metrics, hyperparameters, and system resource usage during your AI training.
    *   Supports cloud and offline usage.
    *   Records hyperparameters, metrics summaries, and table analysis.
    *   Visualizes training progress through an intuitive UI.
    *   Supports various metadata types including scalar metrics, images, audio, text, videos, 3D point clouds, biochemical molecules, and custom ECharts charts.
    *   Offers diverse chart types: Line charts, media charts (images, audio, text, video), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, and custom charts.
    *   LLM-generated content visualization components with Markdown rendering.
    *   Automated background recording of logs, hardware environment, Git repository, Python environment, Python library list, and project runtime directory.
    *   Resume training and add new metric data after training completion or interruption.
*   **⚡️ Extensive Framework Integrations:**  Seamlessly integrate with over 30 popular frameworks.
*   **💻 Hardware Monitoring:**  Monitor and record CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (MX), Moore Threads GPU, memory, disk, and network metrics in real time.
*   **📦 Experiment Management:** Manage multiple projects and experiments through a centralized dashboard.
*   **🆚 Result Comparison:** Compare hyperparameters and results of different experiments using online tables and comparison charts.
*   **👥 Collaborative Features:**  Collaborate with your team by sharing and syncing experiments within a project.
*   **✉️ Share Results:** Share your experiments with persistent URLs for easy access.
*   **🔌 Plugin Extensibility:** Extend SwanLab's functionality with plugins for notifications and more.
*   **💻 Self-Hosting Support:**  Use SwanLab in offline environments with a self-hosted community version.

> \[!IMPORTANT]
>
> **Star the project** to receive notifications of all releases! ⭐️

![star-us](readme_files/star-us.png)

<br/>

## Installation

```bash
pip install swanlab
```

## Usage
```python
import swanlab

# Initialize a new swanlab experiment
swanlab.init(
    project="my-first-ml",
    config={'learning-rate': 0.003},
)

# Record metrics
for i in range(10):
    swanlab.log({"loss": i, "acc": i})
```

## Additional Information

*   **[Online Demos](#-在线演示)**: View interactive demos showcasing SwanLab's capabilities.
*   **[Self-Hosting](#-自托管)**: Deploy your own instance for offline use.
*   **[Real-World Examples](#-实战案例)**: Explore tutorials and projects that utilize SwanLab.
*   **[Hardware Monitoring](#-硬件记录)**: See the full list of supported hardware.
*   **[Framework Integrations](#-框架集成)**: Learn more about supported frameworks.
*   **[Plugins and API](#-插件与api)**: Extend and customize SwanLab.
*   **[Comparisons](#-🆚-与熟悉的工具的比较)**: Compare SwanLab to Tensorboard and Weights & Biases.
*   **[Community](#-社区)**: Find resources for support and collaboration.
*   **[Contributing](#-为-swanlab-做出贡献)**:  Get involved and help improve SwanLab.

[GitHub Repo: https://github.com/SwanHubX/SwanLab](https://github.com/SwanHubX/SwanLab)