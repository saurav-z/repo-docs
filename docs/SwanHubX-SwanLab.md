<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## 🚀 SwanLab: Your Open-Source AI Training Tracker and Visualizer

**SwanLab** is an open-source, user-friendly platform designed to track, visualize, and collaborate on your machine learning experiments. It provides an intuitive way to monitor your training process, analyze results, and share insights. [Visit the original repository](https://github.com/SwanHubX/SwanLab) for more details.

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

<img src="readme_files/swanlab-overview.png">

[中文](README.md) / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking:** Easily track metrics, hyperparameters, and model artifacts.
*   **Rich Visualization:**  Visualize training progress with interactive charts and dashboards.
*   **Flexible Framework Integrations:** Supports 30+ popular deep learning frameworks, including PyTorch, TensorFlow, Hugging Face, and more.
*   **Hardware Monitoring:** Monitor CPU, GPU, memory, and other hardware resources in real time.
*   **Collaboration Tools:** Facilitate team collaboration with shared experiments and project management.
*   **Self-Hosting:**  Deploy SwanLab on-premise for complete control over your data.
*   **Extensible with Plugins:**  Customize SwanLab with plugins for notifications, data logging, and more.
*   **Comprehensive Data Types:**  Supports scalar metrics, images, audio, text, video, 3D point clouds, and more.

## Getting Started

### Installation

```bash
pip install swanlab
```

### Quick Start

```python
import swanlab

# Initialize a new SwanLab experiment
swanlab.init(
    project="my-first-ml",
    config={'learning-rate': 0.003},
)

# Log metrics
for i in range(10):
    swanlab.log({"loss": i, "acc": i})
```

## Resources

*   [Online Demo](https://swanlab.cn)
*   [Documentation](https://docs.swanlab.cn)
*   [Community](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
*   [GitHub Repository](https://github.com/SwanHubX/SwanLab)
*   [Self-Hosting Guide](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

<br>
<div align="center">
<img src="./readme_files/swanlab-and-user.png" width="50%" />
</div>