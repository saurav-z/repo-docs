<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## SwanLab: The Open-Source Deep Learning Training Tracker and Visualizer

**SwanLab streamlines your deep learning workflow by providing an open-source, modern, and easy-to-integrate platform for tracking, visualizing, and collaborating on your machine learning experiments.**  Seamlessly integrated with 30+ popular frameworks and supporting both cloud and local use, SwanLab empowers researchers to understand, refine, and share their AI models.

<a href="https://swanlab.cn">🔥SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Documentation</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks</a>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)


![](readme_files/swanlab-overview.png)

中文 / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking & Visualization**:
    *   Track and visualize key metrics, hyperparameters, and metadata.
    *   Rich UI for interactive exploration of your training runs.
    *   Comprehensive data types supported: scalar metrics, images, audio, text, videos, 3D point clouds, biochemical molecules, and custom ECharts.
    *   Interactive chart types: line charts, media (image, audio, text, video), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts and custom charts.
    *   LLM output visualization component with Markdown rendering for text data.

*   **Framework Integrations**:
    *   Seamless integration with 30+ popular deep learning frameworks, including PyTorch, TensorFlow, Hugging Face Transformers, and more.
    *   Supports advanced features like distributed training and hyperparameter optimization.

*   **Hardware Monitoring**:
    *   Real-time monitoring of CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (Muxi), Moore Threads GPU, and memory usage.

*   **Experiment Management**:
    *   Centralized dashboard for managing projects and experiments.
    *   Project-level and experiment-level organization.

*   **Comparison & Analysis**:
    *   Compare experiment results side-by-side using tables and interactive charts.
    *   Identify trends and insights across different runs.

*   **Collaboration**:
    *   Share and collaborate on experiments with your team.
    *   Real-time synchronization of experiments within a project.

*   **Shareability**:
    *   Generate shareable URLs for individual experiments.
    *   Embed experiments in reports and presentations.

*   **Self-Hosting**:
    *   Supports offline use and self-hosting for local experimentation and private deployments.

*   **Plugin Extensibility**:
    *   Extend functionality through plugins: Notifications (Slack, Discord, Email, etc.) and Custom data writers

<br/>

## Table of Contents

*   [🌟 Recent Updates](#-recent-updates)
*   [👋🏻 What is SwanLab?](#-what-is-swanlab)
*   [📃 Online Demo](#-online-demo)
*   [🏁 Quickstart](#-quickstart)
*   [💻 Self-Hosting](#-self-hosting)
*   [🔥 Real-World Examples](#-real-world-examples)
*   [🎮 Hardware Monitoring](#-hardware-monitoring)
*   [🚗 Framework Integrations](#-framework-integrations)
*   [🔌 Plugins and API](#-plugins-and-api)
*   [🆚 Comparison with Similar Tools](#-comparison-with-similar-tools)
*   [👥 Community](#-community)
*   [📃 License](#-license)

<br/>

## 🌟 Recent Updates

*   ... (Recent Updates - condensed, focusing on key features & benefits.  Link to changelog.)

<details><summary>Full Changelog</summary>
... (Full Changelog - keep it in a details section for those interested, but not the main focus)
</details>

<br>

## 👋🏻 What is SwanLab

SwanLab is an open-source, lightweight AI model training tracker and visualizer, providing a platform to track, record, compare, and collaborate on experiments.

SwanLab is designed for AI researchers, with a user-friendly Python API and a beautiful UI. It offers features such as **training visualization, automatic logging, hyperparameter recording, experiment comparison, and multi-person collaboration**. SwanLab enables researchers to find training problems based on intuitive visual charts, compare multiple experiments to gain research inspiration, and break down communication barriers in the team through **online web pages** and **multi-person collaborative training** based on the organization, improving organizational training efficiency.

https://github.com/user-attachments/assets/7965fec4-c8b0-4956-803d-dbf177b44f54

Key features include:

*   **Experiment Tracking & Visualization**: Track and visualize metrics, hyperparameters, and model outputs.
*   **Framework Integrations**: Supports 30+ popular frameworks.
*   **Hardware Monitoring**: Real-time monitoring of system resource usage.
*   **Experiment Management**: Centralized dashboard for organizing and managing experiments.
*   **Result Comparison**: Compare results using interactive charts and tables.
*   **Collaboration**: Facilitate team-based training with real-time updates and sharing.
*   **Shareability**: Share experiments easily with shareable URLs.
*   **Self-Hosting**:  Supports offline use and self-hosting.
*   **Plugin Extensibility**: Enhance the functionality with plugins (notifications, writers, etc.)

> \[!IMPORTANT]
>
> **Star the project on GitHub** to get notified of new releases! ⭐️

![star-us](readme_files/star-us.png)

<br>

## 📃 Online Demo

Explore the interactive capabilities of SwanLab with our online demos:

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Image classification on the cat/dog dataset with a ResNet50 model | Object detection on the COCO128 dataset with YOLOv8 |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Instruction fine-tuning of a Qwen2 language model. |  Stock prediction with a simple LSTM model. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Audio classification experiments on a ResNeXt101 model |  Fine-tuning on the COCO2014 dataset based on the Qwen2-VL multi-modal large model |

| [EasyR1 Multi-Modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Training multi-modal LLM RL using the EasyR1 framework | GRPO training with the Qwen2.5-0.5B model on the GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

... (Source install instructions)

</details>

<details><summary>Offline Dashboard Installation</summary>

... (offline board install instructions)

</details>

### 2.  Login and get your API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Login and copy your API Key from the user settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and run:

```bash
swanlab login
```

Enter your API Key when prompted.

### 3. Integrate SwanLab into your code

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

That's it!  Visit [SwanLab](https://swanlab.cn) to view your first experiment.

<br>

## 💻 Self-Hosting

Self-hosting allows you to use the SwanLab dashboard in an offline environment.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy using Docker

... (Docker Deployment Instructions)

### 2. Connect Experiments to Self-Hosted Instance

... (Login for self-hosted deployment instructions)

<br>

## 🔥 Real-World Examples

**Open-source projects using SwanLab:**
*   [happy-llm](https://github.com/datawhalechina/happy-llm): Tutorial on LLM principles and practice.  ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm): Tutorial for finetuning and deploying open-source LLMs and MLLMs.  ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series explanation, expansion, and reproduction.  ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)

**Published Papers using SwanLab:**
*   ... (Paper examples)

**Tutorials:**
*   ... (tutorial examples)

🌟  Submit a PR with your tutorial if you want to be featured!

<br>

## 🎮 Hardware Monitoring

SwanLab automatically logs and monitors hardware information and resource utilization.

... (Hardware monitoring table.)

... (Add a sentence on how to request support for additional hardware.)

<br>

## 🚗 Framework Integrations

Use SwanLab with your favorite frameworks! (List integrations and link to docs.)

... (Framework integration list.)

<br>

## 🔌 Plugins and API

Extend SwanLab's functionality!

... (plugins and API info and links)

<br>

## 🆚 Comparison with Similar Tools

### TensorBoard vs SwanLab

*   **Cloud Support:** SwanLab provides easy cloud synchronization.
*   **Collaboration:** SwanLab excels in multi-user collaboration.
*   **Centralized Dashboard**:  SwanLab provides a unified dashboard.
*   **Powerful Tables**:  SwanLab has powerful tables.

### Weights & Biases vs SwanLab

* Weights & Biases is a closed-source, network-dependent MLOps platform
* SwanLab is open-source, free, and has a self-hosted version

<br>

## 👥 Community

### Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Documentation.
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline Dashboard code.
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting scripts.

### Community and Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): For bugs and questions.
*   [Email Support](zeyi.lin@swanhub.co): for feedback.
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Discuss issues and share AI technologies.

### SwanLab README Badges

Add the SwanLab badge to your README:

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

Find more design assets: [assets](https://github.com/SwanHubX/assets)

### Cite SwanLab

If SwanLab has helped your research, please consider citing it:

```bibtex
@software{Zeyilin_SwanLab_2023,
  author = {Zeyi Lin, Shaohong Chen, Kang Li, Qiushan Jiang, Zirui Cai,  Kaifang Ji and {The SwanLab team}},
  doi = {10.5281/zenodo.11100550},
  license = {Apache-2.0},
  title = {{SwanLab}},
  url = {https://github.com/swanhubx/swanlab},
  year = {2023}
}
```

### Contributing to SwanLab

Read the [Contributing Guide](CONTRIBUTING.md).

Thank you for supporting SwanLab!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## 📃 License

This repository is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).