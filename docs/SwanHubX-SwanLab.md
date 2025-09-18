<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## SwanLab: The Open-Source AI Training Tracker and Visualizer

**SwanLab is an open-source, modern tool designed to streamline deep learning training, offering a centralized platform for experiment tracking, visualization, and collaboration.** It seamlessly integrates with 30+ popular machine learning frameworks, providing both cloud and offline capabilities, making it easier than ever to monitor, analyze, and share your AI training progress. Check out the original repo [here](https://github.com/SwanHubX/SwanLab).

<p align="center">
  <a href="https://swanlab.cn">🔥 SwanLab Online</a> &nbsp; | &nbsp;
  <a href="https://docs.swanlab.cn">📃 Documentation</a> &nbsp; | &nbsp;
  <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> &nbsp; | &nbsp;
  <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> &nbsp; | &nbsp;
  <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> &nbsp; | &nbsp;
  <a href="https://swanlab.cn/benchmarks">Benchmarks</a>
</p>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![Wechat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

<br/>
<div align="center">
  <img src="readme_files/swanlab-overview.png" alt="SwanLab Overview" width="80%">
</div>

<br/>

**Explore the Documentation:**
*   [中文](README.md) / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

<p align="center">
  👋 Join our <a href="https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html">WeChat Group</a>
</p>

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Table of Contents

-   [🚀 Key Features](#-key-features)
-   [📃 Online Demo](#-online-demo)
-   [🏁 Quick Start](#-quick-start)
-   [💻 Self-Hosting](#-self-hosting)
-   [🔥 Practical Examples](#-practical-examples)
-   [🎮 Hardware Monitoring](#-hardware-monitoring)
-   [🚗 Framework Integrations](#-framework-integrations)
-   [🔌 Plugins and API](#-plugins-and-api)
-   [🆚 Comparison with Similar Tools](#-comparison-with-similar-tools)
-   [👥 Community](#-community)
-   [📃 License](#-license)
-   [Star History](#star-history)

<br/>

## 🚀 Key Features

SwanLab simplifies and enhances your machine learning workflow with these core functionalities:

*   **📊 Experiment Tracking & Parameter Logging**: Effortlessly integrate with your code to track key metrics, hyperparameters, and experiment summaries.
    *   ☁️ **Cloud-Based Accessibility**: Access your training progress anytime, anywhere, with cloud support.
    *   📝 **Comprehensive Data Recording**: Log hyperparameters, summarize metrics, and analyze results in tables.
    *   🌸 **Interactive Visualizations**: Gain insights through intuitive UI visualizations of your experiments.

*   **📦 Support for Various Data Types**: Visualize a wide array of data types, including scalar metrics, images, audio, text, videos, 3D point clouds, and custom ECharts charts.
    *   **Customizable Charts**: Create flexible and interactive visualizations using [SwanLab's ECharts integration](https://docs.swanlab.cn/guide_cloud/experiment_track/log-custom-chart.html)
    *   **LLM Content Visualization**: Display text content with Markdown rendering, specifically for large language model training.

    ![text-chart](readme_files/text-chart.gif)

    *   **Automatic Background Recording**: Log logging, hardware environment, Git repository details, Python environment, library listings, and project directory.
    *   **Resume Support**: Support continuous training by logging new metric data into the same experiment after training is completed or interrupted.

*   **⚡️ Extensive Framework Integration**: Seamless integration with over 30 popular ML frameworks, including PyTorch, 🤗HuggingFace Transformers, PyTorch Lightning, and many more (See Framework Integrations Section).

    ![](readme_files/integrations.png)

*   **💻 Hardware Monitoring**: Monitor system-level hardware metrics in real-time, including CPU, GPU (Nvidia, Ascend, etc.), MLU (Cambricon), and memory usage.

*   **📦 Experiment Management**: Centralized dashboards for efficient management of multiple projects and experiments.

*   **🆚 Results Comparison**: Analyze and compare experiment results through online tables and charts to spark innovation.

    ![](readme_files/swanlab-table.png)

*   **👥 Online Collaboration**: Enable team-based training with real-time synchronization, sharing of experiment links, and collaborative discussions.

*   **✉️ Shareable Results**: Generate and share persistent URLs for individual experiments, for easy sharing with collaborators or embedding in online documents.

*   **💻 Self-Hosting Support**: Use SwanLab in offline environments with self-hosting capabilities, enabling experiment management and dashboard viewing (See Self-Hosting Section).

*   **🔌 Plugin Extensibility**: Extend SwanLab's capabilities with plugins for notifications, logging, and custom features (See Plugins and API Section).

> \[!IMPORTANT]
>
> **Star the project** to receive all release notifications directly from GitHub! ⭐️

![star-us](readme_files/star-us.png)

<br>

## 📃 Online Demo

Explore SwanLab through interactive online demos, each showcasing different applications:

| [ResNet50 Cat vs. Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Track image classification using a simple ResNet50 model on a cats and dogs dataset. | Track training hyperparameters and metrics using Yolov8 on the COCO128 dataset. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Track instruction fine-tuning training of Qwen2 large language models. | Using a simple LSTM model on the Google stock price dataset to predict future stock prices. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Experiment tracking of progress from ResNet to ResNeXt on audio classification tasks. | Lora fine-tuning the Qwen2-VL multi-modal large model on COCO2014 dataset. |

| [EasyR1 Multi-Modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Multi-modal LLM RL training using the EasyR1 framework. | GRPO training on the GSM8k dataset based on the Qwen2.5-0.5B model |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

To experience the latest features, install from source.

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Extension Installation</summary>

[Offline Dashboard Documentation](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get Your API Key

1.  [Register a free account](https://swanlab.cn)
2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and type:

```bash
swanlab login
```

Then, enter your API Key when prompted and press Enter.

### 3. Integrate SwanLab into your Code

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

You're all set! Go to [SwanLab](https://swanlab.cn) to view your first SwanLab experiment.

<br>

## 💻 Self-Hosting

The self-hosted community edition allows for offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted Version with Docker

Refer to the documentation for details: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Quick Installation for China:

```bash
./install.sh
```

Install by pulling images from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Specify Experiments to Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Once logged in, you can log your experiments to the self-hosted service.

<br>

## 🔥 Practical Examples

**Great tutorial open source projects that use SwanLab:**
-   [happy-llm](https://github.com/datawhalechina/happy-llm): Tutorial on the principles and practice of large language models ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
-   [self-llm](https://github.com/datawhalechina/self-llm): "Open Source LLM Guide" tailored for Chinese developers, teaching quick fine-tuning (full/Lora), deployment of domestic/international open source LLMs/multi-modal LLMs in a Linux environment. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
-   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, extension, and reproduction ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
-   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): Fine-tuning that combines the visual head of SmolVLM2 with the Qwen3-0.6B model. ![GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)
-   [OPPO/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models): End-to-end Agent foundation models through multi-Agent distillation and Agent RL. ![GitHub Repo stars](https://img.shields.io/github/stars/OPPO-PersonalAI/Agent_Foundation_Models)

**Excellent papers using SwanLab:**
-   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
-   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
-   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
-   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorial Articles:**
-   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
-   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
-   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
-   [Resnet Cat vs. Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
-   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
-   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
-   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
-   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
-   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
-   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
-   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
-   [LLM Pre-training](https://docs.swanlab.cn/examples/pretrain_llm.html)
-   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
-   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
-   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
-   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
-   [Qwen2-VL Multi-modal Large Model Fine-tuning Practice](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
-   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
-   [Qwen3-SmVL-0.6B Multi-modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
-   [LeRobot Embodied Intelligence Tutorial](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
-   [GLM-4.5-Air-LoRA and SwanLab Visualization](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
-   [How to do RAG? The SwanLab Document Assistant Solution is Open Sourced](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟 We welcome PRs to add tutorials if you want to include them!

<br>

## 🎮 Hardware Monitoring

SwanLab records hardware information and resource usage during AI training. Below is the support table:

| Hardware | Information Recording | Resource Monitoring | Script |
| --- | --- | --- | --- |
| NVIDIA GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU | ✅ | ✅ | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC | ✅ | ✅ | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU | ✅ | ✅ | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅ | ✅ | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU | ✅ | ✅ | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory | ✅ | ✅ | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk | ✅ | ✅ | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

If you want to record other hardware, welcome to submit an Issue and PR!

<br>

## 🚗 Framework Integrations

Use SwanLab with your favorite frameworks!  
Here's a list of the frameworks we have integrated with. Welcome to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to give feedback on the framework you want to integrate.

**Basic Frameworks**
-   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
-   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
-   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized / Fine-tuning Frameworks**
-   [PyTorch Lightning](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-lightning.html)
-   [HuggingFace Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-transformers.html)
-   [LLaMA Factory](https://docs.swanlab.cn/guide_cloud/integration/integration-llama-factory.html)
-   [Modelscope Swift](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html)
-   [DiffSynth Studio](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html)
-   [Sentence Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-sentence-transformers.html)
-   [PaddleNLP](https://docs.swanlab.cn/guide_cloud/integration/integration-paddlenlp.html)
-   [OpenMind](https://modelers.cn/docs/zh/openmind-library/1.0.0/basic_tutorial/finetune/finetune_pt.html#%E8%AE%AD%E7%BB%83%E7%9B%91%E6%8E%A7)
-   [Torchtune](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-torchtune.html)
-   [XTuner](https://docs.swanlab.cn/guide_cloud/integration/integration-xtuner.html)
-   [MMEngine](https://docs.swanlab.cn/guide_cloud/integration/integration-mmengine.html)
-   [FastAI](https://docs.swanlab.cn/guide_cloud/integration/integration-fastai.html)
-   [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html)
-   [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html)
-   [MLX-LM](https://docs.swanlab.cn/guide_cloud/integration/integration-mlx-lm.html)

**Evaluation Frameworks**
-   [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)

**Computer Vision**
-   [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
-   [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
-   [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
-   [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
-   [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)

**Reinforcement Learning**
-   [Stable Baseline3](https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html)
-   [veRL](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)
-   [HuggingFace trl](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-trl.html)
-   [EasyR1](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
-   [AReaL](https://docs.swanlab.cn/guide_cloud/integration/integration-areal.html)
-   [ROLL](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

**Other Frameworks：**
-   [Tensorboard](https://docs.swanlab.cn/guide_cloud/integration/integration-tensorboard.html)
-   [Weights&Biases](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html)
-   [MLFlow](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html)
-   [HuggingFace Accelerate](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html)
-   [Ray](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html)
-   [Unsloth](https://docs.swanlab.cn/guide_cloud/integration/integration-unsloth.html)
-   [Hydra](https://docs.swanlab.cn/guide_cloud/integration/integration-hydra.html)
-   [Omegaconf](https://docs.swanlab.cn/guide_cloud/integration/integration-omegaconf.html)
-   [OpenAI](https://docs.swanlab.cn/guide_cloud/integration/integration-openai.html)
-   [ZhipuAI](https://docs.swanlab.cn/guide_cloud/integration/integration-zhipuai.html)

[More Integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br>

## 🔌 Plugins and API

Extend SwanLab's capabilities through plugins, enhancing your experiment management experience!

-   [Customize your plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email Notification](https://docs.swanlab.cn/plugin/notification-email.html)
-   [Feishu Notification](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notification](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [WeChat Work Notification](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord Notification](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack Notification](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV Recorder](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [File Logdir Recorder](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI:
-   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparison with Similar Tools

### Tensorboard vs SwanLab

-   **☁️ Online Support**:
    SwanLab simplifies the synchronization and saving of experiments in the cloud, allowing you to remotely view training progress, manage historical projects, share experiment links, send real-time message notifications, and view experiments on multiple devices. Tensorboard is an offline experiment tracking tool.
-   **👥 Collaboration**:
    SwanLab easily manages multi-person training projects, shares experiment links, and enables cross-space communication and discussion. Tensorboard is mainly designed for individuals and is difficult to conduct multi-person collaboration and experiment sharing.
-   **💻 Persistent, Centralized Dashboard**:
    Your results are recorded in the same centralized dashboard regardless of where you train your model, whether on a local computer, in a lab cluster, or on a public cloud GPU instance. Using TensorBoard, it takes time to copy and manage TFEvent files from different machines.
-   **💪 More Powerful Tables**:
    You can view, search, and filter results from different experiments through the SwanLab table, which makes it easy to view thousands of model versions and find the best-performing model for different tasks. TensorBoard is not suitable for large projects.

### Weights and Biases vs SwanLab

-   Weights and Biases is a closed-source MLOps platform that must be connected to the network.
-   SwanLab supports not only network usage but also open source, free, and self-hosted versions.

<br>

## 👥 Community

### Peripheral Repositories

-   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosted deployment scripts repository
-   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation repository
-   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard repository, storing the web code for the lightweight offline dashboard opened by `swanlab watch`

### Community and Support

-   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Errors and problems encountered when using SwanLab
-   [Email Support](zeyi.lin@swanhub.co): Feedback about issues using SwanLab
-   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Discuss issues using SwanLab and share the latest AI technologies

### SwanLab README Badges

Add the SwanLab badge to your README if you like using SwanLab:

[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design resources: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Papers

If you find SwanLab helpful in your research, please consider citing it in the following format:

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

Want to contribute to SwanLab?  Please start by reading the [Contribution Guide](CONTRIBUTING.md).

We are very welcome to support SwanLab through sharing on social media, events, and conferences, thank you very much!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## 📃 License

This repository is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star