<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

**Supercharge your AI experiments with SwanLab: the open-source, modern platform for tracking, visualizing, and collaborating on your deep learning projects.**

<a href="https://swanlab.cn">🔥SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Documentation</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Community</a>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
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

## Key Features:

*   **Experiment Tracking & Visualization:** Track metrics, visualize training progress with intuitive charts, and gain insights into your model's performance.
*   **Automatic Logging:** Automatically log hyperparameters, system metrics (CPU, GPU, etc.), and more.
*   **Rich Data Types:** Support for scalar metrics, images, audio, text, video, 3D point clouds, custom Echarts, and more.
*   **Flexible Charting:** Comprehensive charting capabilities, including line charts, media visualization, 3D point clouds, and custom charts.
*   **Framework Integrations:** Seamless integration with popular frameworks such as PyTorch, Hugging Face Transformers, and more (30+ supported).
*   **Hardware Monitoring:** Real-time monitoring of CPU, GPU (Nvidia, Ascend, etc.), and other hardware resources.
*   **Experiment Management:**  Centralized dashboard for managing projects and experiments.
*   **Collaboration:**  Share experiments with your team for collaborative training and result analysis.
*   **Self-Hosting:**  Use SwanLab offline or deploy it on your own server.
*   **Plugin Ecosystem:** Extend SwanLab with plugins for notifications, data writing, and more.

## Table of Contents

-   [What is SwanLab?](#-what-is-swanlab)
-   [Key Features](#key-features)
-   [Getting Started](#-getting-started)
    *   [Installation](#1-installation)
    *   [Login](#2-login-and-get-api-key)
    *   [Integration](#3-integrate-swanlab-with-your-code)
-   [Self-Hosting](#-self-hosting)
    *   [Docker Deployment](#1-deploy-with-docker)
    *   [Connect to Self-Hosted Service](#2-connect-to-self-hosted-service)
-   [Real-World Examples](#-real-world-examples)
-   [Hardware Monitoring Details](#-hardware-monitoring)
-   [Framework Integrations](#-framework-integrations)
-   [Plugins and API](#-plugins-and-api)
-   [Comparison with Similar Tools](#-comparison-with-similar-tools)
-   [Community](#-community)
    *   [Repositories](#周边仓库)
    *   [Support](#社区与支持)
    *   [Badges and Citations](#swanlab-readme-徽章)
    *   [Contributing](#为-swanlab-做出贡献)
-   [License](#-license)
-   [Star History](#star-history)

<br/>

## 🚀 What is SwanLab?

SwanLab is an open-source, user-friendly platform designed for tracking, visualizing, and collaborating on deep learning experiments. It provides researchers and developers with a powerful suite of tools to monitor and analyze their training runs effectively. SwanLab makes it easy to compare experiments, identify performance bottlenecks, and share results with your team, ultimately accelerating your AI development workflow.

<br/>

## 🏁 Getting Started

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
cd SwanLab
pip install -e .  # Or `pip install -e .`
```

</details>

<details><summary>Install Dashboard</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Log in and go to User Settings > [API Key](https://swanlab.cn/settings) to get your API key.
3.  Open your terminal and run:

```bash
swanlab login
```

Enter your API key when prompted.

### 3. Integrate SwanLab with Your Code

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

Check out your experiment at [SwanLab](https://swanlab.cn).

<br/>

## 💻 Self-Hosting

SwanLab offers a self-hosted community version for offline use and data privacy.

### 1. Deploy with Docker

Follow the steps in the [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker

# For China:
./install.sh

# From DockerHub:
./install-dockerhub.sh
```

### 2. Connect to Self-Hosted Service

```bash
swanlab login --host http://localhost:8000
```

Now, your experiments will be recorded to your self-hosted instance.

<br/>

## 🔥 Real-World Examples

Explore example projects and tutorials:

*   [ResNet50 Cat/Dog Classification][demo-cats-dogs]
*   [Yolov8 Object Detection - COCO128][demo-yolo]
*   [Qwen2 Instruction Fine-tuning][demo-qwen2-sft]
*   [LSTM Google Stock Prediction][demo-google-stock]
*   [ResNeXt101 Audio Classification][demo-audio-classification]
*   [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl]
*   [EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl]
*   [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo]

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br/>

## 🎮 Hardware Monitoring

SwanLab monitors hardware information and resource usage.

| Hardware          | Info Recording | Resource Monitoring | Script                                                               |
| :---------------- | :-------------: | :-----------------: | :------------------------------------------------------------------- |
| NVIDIA GPU        |       ✅        |         ✅          | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)   |
| Ascend NPU        |       ✅        |         ✅          | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)   |
| Apple SOC         |       ✅        |         ✅          | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)    |
| Cambricon MLU     |       ✅        |         ✅          | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU     |       ✅        |         ✅          | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU |       ✅        |         ✅          | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU         |       ✅        |         ✅          | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py)    |
| Hygon DCU         |       ✅        |         ✅          | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py)    |
| CPU               |       ✅        |         ✅          | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)          |
| Memory            |       ✅        |         ✅          | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)       |
| Disk              |       ✅        |         ✅          | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)          |
| Network           |       ✅        |         ✅          | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)    |

Submit an [Issue](https://github.com/swanhubx/swanlab/issues) or [PR](https://github.com/swanhubx/swanlab/pulls) if you want to record other hardware.

<br/>

## 🚗 Framework Integrations

Easily integrate SwanLab with your favorite frameworks!

**Base Frameworks**

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Finetuning Frameworks**

*   [PyTorch Lightning](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-lightning.html)
*   [HuggingFace Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-transformers.html)
*   [LLaMA Factory](https://docs.swanlab.cn/guide_cloud/integration/integration-llama-factory.html)
*   [Modelscope Swift](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html)
*   [DiffSynth Studio](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html)
*   [Sentence Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-sentence-transformers.html)
*   [PaddleNLP](https://docs.swanlab.cn/guide_cloud/integration/integration-paddlenlp.html)
*   [OpenMind](https://modelers.cn/docs/zh/openmind-library/1.0.0/basic_tutorial/finetune/finetune_pt.html#%E8%AE%AD%E7%BB%83%E7%9B%91%E6%8E%A7)
*   [Torchtune](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-torchtune.html)
*   [XTuner](https://docs.swanlab.cn/guide_cloud/integration/integration-xtuner.html)
*   [MMEngine](https://docs.swanlab.cn/guide_cloud/integration/integration-mmengine.html)
*   [FastAI](https://docs.swanlab.cn/guide_cloud/integration/integration-fastai.html)
*   [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html)
*   [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html)

**Evaluation Frameworks**

*   [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)

**Computer Vision**

*   [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
*   [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
*   [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
*   [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
*   [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)

**Reinforcement Learning**

*   [Stable Baseline3](https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html)
*   [veRL](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)
*   [HuggingFace trl](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-trl.html)
*   [EasyR1](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
*   [AReaL](https://docs.swanlab.cn/guide_cloud/integration/integration-areal.html)
*   [ROLL](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

**Other Frameworks**

*   [Tensorboard](https://docs.swanlab.cn/guide_cloud/integration/integration-tensorboard.html)
*   [Weights&Biases](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html)
*   [MLFlow](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html)
*   [HuggingFace Accelerate](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html)
*   [Ray](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html)
*   [Unsloth](https://docs.swanlab.cn/guide_cloud/integration/integration-unsloth.html)
*   [Hydra](https://docs.swanlab.cn/guide_cloud/integration/integration-hydra.html)
*   [Omegaconf](https://docs.swanlab.cn/guide_cloud/integration/integration-omegaconf.html)
*   [OpenAI](https://docs.swanlab.cn/guide_cloud/integration/integration-openai.html)
*   [ZhipuAI](https://docs.swanlab.cn/guide_cloud/integration/integration-zhipuai.html)

[More Integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br/>

## 🔌 Plugins and API

Enhance your SwanLab experience with plugins:

*   [Custom Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeCom Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:

*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br/>

## 🆚 Comparison with Similar Tools

### TensorBoard vs SwanLab

*   **Online Support:** SwanLab offers cloud-based sync and storage, allowing for remote monitoring, project management, and easy sharing. TensorBoard is primarily a local tool.
*   **Collaboration:** SwanLab supports collaboration for multi-team projects, experiment sharing, and communication. TensorBoard focuses on individual use.
*   **Dashboard:** SwanLab provides a persistent and centralized dashboard for all your experiments. TensorBoard requires managing TFEvent files from different machines.
*   **Powerful Tables:** SwanLab offers tables to view, search, and filter results from different experiments. TensorBoard is not optimized for large projects.

### Weights and Biases vs SwanLab

*   Weights and Biases is a closed-source, cloud-based MLOps platform.
*   SwanLab is an open-source, free, and self-hostable alternative.

<br/>

## 👥 Community

### Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation.
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline Dashboard.
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting scripts.

### Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report bugs and ask questions.
*   [Email Support](zeyi.lin@swanhub.co): For feedback and questions.
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Discuss SwanLab and share AI insights.

### Badges and Citations

Add a SwanLab badge to your README:

[![Track with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

**Citation:**

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

### Contributing

See the [Contributing Guide](CONTRIBUTING.md) to learn about contributing to SwanLab.  We appreciate your support through social media sharing!

<br/>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br/>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

<br/>

## 📃 License

This project is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)