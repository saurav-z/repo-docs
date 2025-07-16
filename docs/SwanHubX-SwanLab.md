<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-single-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-single.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-single.svg" width="70" height="70">
</picture>

<h1>SwanLab: Unleash the Power of AI Experiment Tracking and Visualization</h1>

**SwanLab is an open-source, modern tool designed to supercharge your deep learning experiments, providing intuitive tracking, visualization, and collaboration features.**  Easily integrate with 30+ major frameworks for seamless cloud/offline use!

<a href="https://swanlab.cn">🔥SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Documentation</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks</a>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)


![](readme_files/swanlab-overview.png)

[中文](README.md) / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>

<br/>

## Key Features

*   **Experiment Tracking:**  Effortlessly track metrics, hyperparameters, and system resources during your training runs.
    *   **Cloud & Offline Support:** Use SwanLab online (like Weights & Biases) or locally.
    *   **Rich Data Types:** Supports scalars, images, audio, text, 3D point clouds, and custom charts.
    *   **Extensive Charting:** Visualize training progress with line charts, media views, 3D visualizations, and custom chart types.
    *   **LLM-Focused Text Visualization:**  Specialized text visualizations with Markdown rendering for LLM experiments.
    *   **Automatic Logging:** Logs model training information, hardware environment, Git repository details, Python environment, and project dependencies automatically.
    *   **Resume Support:** Resume interrupted training runs by appending new metrics to the same experiment.
*   **Framework Integrations:** Compatible with 30+ popular deep learning frameworks, including PyTorch, TensorFlow, Hugging Face Transformers, and more.
*   **Hardware Monitoring:** Real-time monitoring of CPU, GPU (Nvidia, Ascend, Apple, etc.), MLU (Cambricon), Memory, Disk, and Network usage.
*   **Experiment Management:** Centralized dashboard for managing projects and experiments, and quick overviews.
*   **Result Comparison:** Compare hyperparameters and results across experiments to identify insights.
*   **Collaboration:** Facilitate collaborative training within teams.
*   **Shareable Results:**  Easily share experiments with persistent URLs.
*   **Self-Hosting:** Utilize the open-source version for offline use, with the ability to view dashboards and manage experiments [instructions](#-自托管).
*   **Plugin Extensibility:** Extend SwanLab's functionality with plugins, such as [Slack notifications](https://docs.swanlab.cn/plugin/notification-slack.html), [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html) and more.

> \[!IMPORTANT]
>
> **Star the project** to stay updated on new releases! ⭐️

![star-us](readme_files/star-us.png)

<br>

## Table of Contents

*   [🌟 Recent Updates](#-最近更新)
*   [👋🏻 What is SwanLab?](#-什么是swanlab)
*   [📃 Online Demo](#-在线演示)
*   [🏁 Quick Start](#-快速开始)
*   [💻 Self-Hosting](#-自托管)
*   [🔥 Case Studies](#-实战案例)
*   [🎮 Hardware Monitoring](#-硬件记录)
*   [🚗 Framework Integrations](#-框架集成)
*   [🔌 Plugins and API](#-插件与api)
*   [🆚 Comparison with Similar Tools](#-与熟悉的工具的比较)
*   [👥 Community](#-社区)
*   [📃 License](#-协议)

<br/>

## 🌟 Recent Updates

-   **(Most Recent Updates are here)**
    - 2025.07.10：📚 More powerful **text view**, supporting Markdown rendering and arrow key switching, can be created by `swanlab.echarts.table` and `swanlab.Text`, [Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)
    - 2025.07.06：🚄 Support **resume training with breakpoints**; new plugin **file recorder**; integration of [ray](https://github.com/ray-project/ray) framework, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); integration of [ROLL](https://github.com/volcengine/ROLL) framework, thanks to [@PanAndy](https://github.com/PanAndy), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

-   **(See full changelog in the original README for details)**
    ...

<details><summary>Full Changelog</summary>
...(content of the full changelog from the original README)
</details>

<br>

## 📃 Online Demo

Explore live demos of SwanLab in action:

| [ResNet50 Cats vs. Dogs][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![Cats vs Dogs][demo-cats-dogs-image]][demo-cats-dogs] | [![YOLO][demo-yolo-image]][demo-yolo] |
| Track a simple ResNet50 model's image classification on a cat and dog dataset. | Track training hyperparameters and metrics using Yolov8 on the COCO128 dataset for object detection. |

| [Qwen2 Instruction Finetuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![Qwen2 Finetuning][demo-qwen2-sft-image]][demo-qwen2-sft] | [![LSTM Stock][demo-google-stock-image]][demo-google-stock] |
| Track Qwen2 large language model's instruction finetuning training, complete simple instruction following. | Train a simple LSTM model on the Google stock dataset to predict future stock prices. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Finetuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![Audio Classification][demo-audio-classification-image]][demo-audio-classification] | [![Qwen2 VL][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiment from ResNet to ResNeXt on audio classification task | LoRA finetuning on Qwen2-VL multi-modal large model on COCO2014 dataset. |

| [EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![EasyR1 RL Training][demo-easyr1-rl-image]][demo-easyr1-rl] | [![Qwen2 GRPO Training][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Train multi-modal LLM RL using EasyR1 framework | GRPO training on Qwen2.5-0.5B model on GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

... (content from original README)

</details>

<details><summary>Offline Dashboard Extension Installation</summary>

... (content from original README)

</details>

### 2. Login and Get Your API Key

1.  [Register for free](https://swanlab.cn)
2.  Log in and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and run:

```bash
swanlab login
```

Enter your API Key when prompted and press Enter.

### 3. Integrate SwanLab into Your Code

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

That's it!  Go to [SwanLab](https://swanlab.cn) to view your experiment.

<br>

## 💻 Self-Hosting

... (content from original README)

<br>

## 🔥 Case Studies

... (content from original README)

<br>

## 🎮 Hardware Monitoring

... (content from original README)

<br>

## 🚗 Framework Integrations

... (content from original README)

<br>

## 🔌 Plugins and API

... (content from original README)

<br>

## 🆚 Comparison with Similar Tools

### Tensorboard vs SwanLab

*   **☁️ Online Support:** SwanLab facilitates online synchronization, storage, and remote access, perfect for monitoring training, managing projects, sharing links, real-time notifications, and multi-device access. Tensorboard is offline.

*   **👥 Collaboration:** SwanLab simplifies collaboration in ML, ideal for managing projects, sharing links, and discussing results in teams. TensorBoard is generally designed for personal use.

*   **💻 Persistent, Centralized Dashboard:** Whether training locally or on cloud GPUs, results are logged to a single dashboard. TensorBoard requires time to copy and manage TFEvent files from various machines.

*   **💪 Enhanced Table:** SwanLab tables offer powerful search, filtering, and comparison features, and easily allow you to find the best-performing models among thousands. TensorBoard is not suitable for large projects.

### Weights and Biases vs SwanLab

... (content from original README)

<br>

## 👥 Community

### Supporting Repos

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs)：Official Documentation Repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard)：Offline Dashboard Code
*   [self-hosted](https://github.com/swanhubx/self-hosted)：Self-hosting Scripts

### Community & Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues)
*   [Email Support](zeyi.lin@swanhub.co)
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>

### SwanLab README Badges

... (content from original README)

### Citing SwanLab in Publications

... (content from original README)

### Contributing

... (content from original README)

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## 📃 License

... (content from original README)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)