<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

**SwanLab: Revolutionizing AI Experiment Tracking and Visualization**

<a href="https://swanlab.cn">🔥SwanLab 在线版</a> · <a href="https://docs.swanlab.cn">📃 文档</a> · <a href="https://github.com/swanhubx/swanlab/issues">报告问题</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">建议反馈</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">更新日志</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">基线社区</a>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

![](readme_files/swanlab-overview.png)

中文 / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 加入我们的[微信群](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>


</div>

<br/>

## Key Features

*   **Easy Integration:** Seamlessly integrate with over 30 popular machine learning frameworks, including PyTorch, TensorFlow, and more.
*   **Real-time Visualization:** Visualize your training progress with intuitive charts and graphs.
*   **Comprehensive Tracking:** Automatically log metrics, hyperparameters, and system information.
*   **Experiment Comparison:**  Easily compare different experiments to identify the best models.
*   **Collaboration & Sharing:** Collaborate with your team and share results through online dashboards.
*   **Hardware Monitoring:** Monitor CPU, GPU, and other hardware resources during training.
*   **Flexible Deployment:** Use SwanLab in the cloud or on-premise for maximum flexibility.
*   **Extensible with Plugins:**  Extend functionality with a range of plugins, including email and Slack notifications.
*   **Rich Data Types:** Supports a wide variety of data types for visualization, including: scalar metrics, images, audio, text, video, 3D point clouds, and more.

<br/>

## Table of Contents

-   [🌟 Recent Updates](#-最近更新)
-   [👋🏻 What is SwanLab?](#-什么是swanlab)
-   [📃 Online Demo](#-在线演示)
-   [🏁 Quick Start](#-快速开始)
-   [💻 Self-Hosting](#-自托管)
-   [🔥 Real-World Examples](#-实战案例)
-   [🎮 Hardware Monitoring](#-硬件记录)
-   [🚗 Framework Integrations](#-框架集成)
-   [🔌 Plugins and API](#-插件与api)
-   [🆚 Comparisons](#-与熟悉的工具的比较)
-   [👥 Community](#-社区)
-   [📃 License](#-协议)

<br/>

## 🌟 Recent Updates

*   **2025.08.19:** 🤔 Improved chart rendering performance, low-invasive loading animations, integration with MLX-LM and SpecForge.
*   **2025.08.06:** 👥 New training collaboration features, including project collaborators and list view for workspaces.
*   **2025.07.29:** 🚀 Experiment filtering and sorting in the sidebar, column control panels for table view, multi-API Key management, new charts (PR curve, ROC curve, confusion matrix)
*   **2025.07.17:** 📊 Enhanced line chart configuration, support for swanlab.Video data type for GIF files, and more.
*   **2025.07.10:** 📚 Improved text view with Markdown rendering and direction key navigation.
*   **2025.07.06:** 🚄 Resume training support, new file recorder plugin, integration with Ray and ROLL frameworks.
*   **2025.06.27:** 📊 Support for local zoom in line charts, and more.
*   **2025.06.20:** 🤗 Integration with Hugging Face Accelerate framework.
*   **2025.06.18:** 🐜 Integration with AReaL framework and enhancements to the experiment sidebar.

<details><summary>Full Changelog</summary>

*   **2025.06.11:** 📊 Support for **swanlab.echarts.table** data type, supports group stretching interaction, and more.
*   **2025.06.08:** ♻️ Support for storing complete experiment log files locally, hardware monitoring support for Hygon DCU
*   **2025.06.01:** 🏸 Support for chart dragging, supports ECharts custom charts, hardware monitoring support for Muxi GPU, integrated PaddleNLP framework
*   **2025.05.25:** Log support for recording standard error streams, hardware monitoring support for Moore Threads.
*   **2025.05.14:** Support for experiment tags, Log Scale for line charts, group dragging, open API.
*   **2025.05.09:** Support for line chart creation and training project GitHub badges.
*   **2025.04.23:** Support line chart editing and Kunlunxin XPU hardware detection.
*   **2025.04.11:** Support line chart partial area selection and hide all chart.
*   **2025.04.08:** Support for swanlab.Molecule data type and improved table view functionality.
*   **2025.04.07:** Integrated with EvalScope for evaluating large model performance.
*   **2025.03.30:** Support swanlab.Settings method and MLU hardware monitoring. Support for Slack and Discord notifications.
*   **2025.03.21:** 🎉🤗HuggingFace Transformers integrated, added Object3D chart, and GPU memory, disk usage, and network monitoring.
*   **2025.03.12:** 🎉🎉SwanLab Private Deployment Version Released!, SwanLab supports plugin extensions.
*   **2025.03.09:** Support for widening the experiment sidebar and added Git code button.
*   **2025.03.06:** Integrated with DiffSynth Studio to track and visualize Diffusion models,
*   **2025.03.04:** Added MLFlow conversion function.
*   **2025.03.01:** Added move experiment function.
*   **2025.02.24:** Integrated with EasyR1 to track and visualize LLM RL experiments.
*   **2025.02.18:** Integrated with Swift to track and visualize LLM finetuning experiments.
*   **2025.02.16:** Added chart moving grouping and creating grouping functions.
*   **2025.02.09:** Integrated with veRL to track and visualize LLM RL experiments.
*   **2025.02.05:** swanlab.log supports nested dictionaries, and more parameters.
*   **2025.01.22:** Added sync_tensorboardX and sync_tensorboard_torch functions
*   **2025.01.17:** Added sync_wandb function, and improved log rendering performance
*   **2025.01.11:** Optimized project table performance and added drag, sort and filter
*   **2025.01.01:** Added line chart persistence smoothing, drag to change size.
*   **2024.12.22:** Integrated with LLaMA Factory to track and visualize LLM finetuning experiments
*   **2024.12.15:** Hardware Monitoring (0.4.0) released, supporting CPU, NPU(Ascend), GPU(Nvidia) information recording and monitoring.
*   **2024.12.06:** Added integration for LightGBM and XGBoost.
*   **2024.11.26:** Hardware section of the environment tab supports the identification of Huawei Ascend NPU and Kunpeng CPU.
</details>

<br>

## 👋🏻 What is SwanLab?

SwanLab is an open-source, modern, and lightweight tool designed for tracking and visualizing machine learning experiments, providing a comprehensive platform for monitoring, logging, comparing, and collaborating on your AI projects.

> \[!IMPORTANT]
>
> **Star the project** to stay updated on releases! ⭐️

![star-us](readme_files/star-us.png)

<br>

## 📃 Online Demo

Explore SwanLab with these interactive demos:

| [ResNet50 Cat vs. Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Tracks a simple ResNet50 model's image classification on the cat and dog dataset. | Tracks training hyperparameters and metrics using Yolov8 on the COCO128 dataset. |

| [Qwen2 Instruction Finetuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Tracks instruction fine-tuning for the Qwen2 large language model. | Predicts future Google stock prices using an LSTM model. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Finetuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Demonstrates the progressive experimentation from ResNet to ResNeXt for audio classification. |  Lora finetuning on the Qwen2-VL multimodal large language model on the COCO2014 dataset. |

| [EasyR1 LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| RL training of multimodal LLM using EasyR1 framework | GRPO training of the Qwen2.5-0.5B model on the GSM8k dataset. |

[More examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

#OR
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Install Offline Dashboard Extension</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get Your API Key

1.  [Register a free account](https://swanlab.cn).

2.  Log in to your account, and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).

3.  Open a terminal and type:

    ```bash
    swanlab login
    ```

    Enter your API Key when prompted, and press Enter to complete the login.

### 3. Integrate SwanLab into your code

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

<br>

## 💻 Self-Hosting

Self-hosting provides an offline dashboard for viewing SwanLab experiments.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted Version with Docker

See the documentation for details: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Fast Installation for China:

```bash
./install.sh
```

Install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point Your Experiments to the Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, you can record your experiments to your self-hosted service.

<br>

## 🔥 Real-World Examples

**Featured Projects using SwanLab:**

-   [happy-llm](https://github.com/datawhalechina/happy-llm): Large language model tutorial
-   [self-llm](https://github.com/datawhalechina/self-llm): A tutorial to fine-tune and deploy large language models
-   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, expansion, and reproduction.
-   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): SmVL2 visual head with Qwen3-0.6B model finetuning.

**Papers using SwanLab:**

-   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
-   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
-   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
-   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorial Articles:**

-   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
-   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
-   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
-   [Resnet Cat vs. Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
-   [YOLO Object Detection](https://docs.swanlab.cn/examples/yolo.html)
-   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
-   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
-   [DQN Reinforcement Learning-Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
-   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
-   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
-   [Stable Diffusion Text-to-Image Finetuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
-   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
-   [GLM4 Instruction Finetuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
-   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
-   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
-   [Qwen3 Medical Model Finetuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
-   [Qwen2-VL Multimodal Large Model Finetuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
-   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
-   [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
-   [LeRobot Embodied Intelligence Tutorial](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
-   [GLM-4.5-Air-LoRA and SwanLab visualization](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
-   [How to do RAG? Open-source SwanLab document assistant solution](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟 Contributions to the tutorial list are welcome!

<br>

## 🎮 Hardware Monitoring

SwanLab monitors hardware information and resource usage during AI training:

| Hardware    | Information Recording | Resource Monitoring | Script                                                 |
| ----------- | --------------------- | ------------------- | ------------------------------------------------------ |
| Nvidia GPU  | ✅                    | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)   |
| Ascend NPU  | ✅                    | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)   |
| Apple SOC   | ✅                    | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)    |
| Cambricon MLU | ✅                    | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅                    | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU| ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU         | ✅                    | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)         |
| Memory      | ✅                    | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)       |
| Disk        | ✅                    | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)         |
| Network     | ✅                    | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)      |

Contributions for additional hardware support are welcome!

<br>

## 🚗 Framework Integrations

Seamlessly integrate SwanLab with your favorite frameworks!

Below is a list of the frameworks we have integrated. Feel free to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to request integrations.

**Core Frameworks**

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Finetuning Frameworks**

*   [PyTorch Lightning](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-lightning.html)
*   [Hugging Face Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-transformers.html)
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
*   [MLX-LM](https://docs.swanlab.cn/guide_cloud/integration/integration-mlx-lm.html)

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

**Other Frameworks:**

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

<br>

## 🔌 Plugins and API

Enhance your experiment tracking with SwanLab plugins!

-   [Create your own Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
-   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [File Log Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:

-   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparisons

### TensorBoard vs. SwanLab

*   **Cloud Support:** SwanLab seamlessly syncs and saves experiments to the cloud, enabling remote access, project management, experiment sharing, and multi-device viewing, while TensorBoard is primarily an offline tool.
*   **Collaboration:** SwanLab facilitates collaboration with features designed for multi-person and cross-team machine learning projects, unlike TensorBoard.
*   **Persistent, Central Dashboard:** Results are logged to a central dashboard, whether training on local machines, clusters, or cloud instances. TensorBoard requires manual TFEvent file management.
*   **Enhanced Tables:** SwanLab's tables support searching, filtering, and comparing results from different experiments. TensorBoard isn't suited for large projects.

### Weights & Biases vs. SwanLab

*   Weights & Biases is a closed-source MLOps platform that requires an internet connection.
*   SwanLab is open-source, free, and supports self-hosting.

<br>

## 👥 Community

### Related Repositories

-   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation repository
-   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard repository
-   [self-hosted](https://github.com/swanhubx/self-hosted): Private deployment script repository

### Community and Support

-   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report issues and ask questions.
-   [Email Support](zeyi.lin@swanhub.co): Provide feedback and ask questions about SwanLab.
-   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discuss and ask questions about using SwanLab.

### SwanLab README Badges

Add these badges to your README to show you use SwanLab:

[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](YOUR_EXPERIMENT_URL)
[![SwanLab Visualize](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](YOUR_EXPERIMENT_URL)

```md
[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](YOUR_EXPERIMENT_URL)
[![SwanLab Visualize](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](YOUR_EXPERIMENT_URL)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Cite SwanLab in your paper

```bibtex
@software{Zeyilin_SwanLab_2023,
  author = {Zeyi Lin, Shaohong Chen, Kang Li, Qiushan Jiang, Zirui Cai,  Kaifang Ji and {The SwanLab team}},
  doi = {10.5281/zenodo.11100550},
  license = {Apache-2.0},
  title = {{SwanLab}},
  url = {https://github.com