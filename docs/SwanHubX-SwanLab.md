<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## SwanLab: AI Experiment Tracking and Visualization

**SwanLab is an open-source, user-friendly tool designed to streamline your deep learning training process with comprehensive experiment tracking, visualization, and collaboration capabilities.**

[🔥SwanLab Online](https://swanlab.cn) | [📃 Documentation](https://docs.swanlab.cn) | [🐛 Report Issues](https://github.com/swanhubx/swanlab/issues) | [💡 Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) | [🗓️ Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) |  <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" />  [基线社区](https://swanlab.cn/benchmarks)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

![](readme_files/swanlab-overview.png)

[中文 / English / 日本語 / Русский](README_EN.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking & Visualization:**
    *   Track key metrics, hyperparameters, and model performance with ease.
    *   Visualize training progress through intuitive UI.
    *   Comprehensive support for various data types: Scalar metrics, images, audio, text, video, 3D point clouds, biochemical molecules, and custom ECharts charts.
    *   Flexible charting options: Line charts, media views (images, audio, text, video), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, and custom charts.

*   **Comprehensive Framework Integration:**
    *   Seamlessly integrates with 30+ popular machine learning frameworks, including PyTorch, Hugging Face Transformers, PyTorch Lightning, LLaMA Factory, and more.
    *   See the full list of integrations [here](#-框架集成).

*   **Hardware Monitoring:**
    *   Real-time monitoring and logging of system-level hardware metrics for CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (MX), Moore Threads GPU, memory, and disk utilization.

*   **Experiment Management & Collaboration:**
    *   Organized dashboard for managing projects and experiments.
    *   Compare results across experiments to gain insights.
    *   Facilitate collaborative training with team members.

*   **Sharing & Self-Hosting:**
    *   Share experiments with shareable URLs.
    *   Supports offline use and self-hosting for complete control and privacy.

*   **Plugin Extensibility:**
    *   Extend SwanLab's capabilities with plugins for notifications, data logging, and more.

## Table of Contents

-   [🌟 Recent Updates](#-最近更新)
-   [👋🏻 What is SwanLab?](#-什么是swanlab)
-   [📃 Online Demo](#-在线演示)
-   [🏁 Quickstart](#-快速开始)
-   [💻 Self-Hosting](#-自托管)
-   [🔥 Real-World Examples](#-实战案例)
-   [🎮 Hardware Monitoring](#-硬件记录)
-   [🚗 Framework Integrations](#-框架集成)
-   [🔌 Plugins & API](#-插件与api)
-   [🆚 Comparison with Similar Tools](#-与熟悉的工具的比较)
-   [👥 Community](#-社区)
-   [📃 License](#-协议)

<br/>

## 🌟 Recent Updates

*   **2025.08.19:** Improved chart rendering performance and low-intrusive loading animation; Integration with excellent MLX-LM and SpecForge frameworks;
*   **2025.08.06:** **Training Collaboration** launched, supports inviting project collaborators; Workspace supports list view and project tag display;
*   **2025.07.29:** Experiment filtering and sorting in sidebar; Table view's column control panel added; **Multi-API Key** management added; New charts, PR, ROC curves;
*   **2025.07.17:** Enhanced line chart configuration options, including line styles, colors, and legends; **swanlab.Video** data type support; Dashboard configuration;
*   **2025.07.10:** **Text View** enhancements with Markdown rendering and arrow key navigation, created by `swanlab.echarts.table` and `swanlab.Text`;
*   **2025.07.06:** Support **resume training**; New **file recorder** plugin; Integration with the [ray](https://github.com/ray-project/ray) and [ROLL](https://github.com/volcengine/ROLL) frameworks;
*   **2025.06.27:** **Line chart zoom**; Support for smoothing individual line charts; Enhanced interaction effects for image charts;
*   **2025.06.20:** Integration with the [accelerate](https://github.com/huggingface/accelerate) framework;
*   **2025.06.18:** Integration with the [AREAL](https://github.com/inclusionAI/AReaL) framework; Mouse hover experiment highlighs and cross-group line comparison;

<details><summary>View Full Changelog</summary>

*   2025.06.11: Added **swanlab.echarts.table** data type; Table view added **max/min value** options;
*   2025.06.08: Supports local storage of full experiment log files via **swanlab sync**; Hardware monitoring for **Hygon DCU**;
*   2025.06.01: Supports **chart dragging**; Supports **ECharts custom charts**, including bar charts, pie charts, histograms, and over 20 chart types; Hardware monitoring supports **MX GPU**; Integration with **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** framework;
*   2025.05.25: Support for logging **standard error streams**, PyTorch Lightning framework printing information recorded better; Hardware monitoring supports **Moore Threads**; Added security protection for running command recording, API Key will be automatically hidden;
*   2025.05.14: Support for **experiment tags**; Support for **Log Scale** for line charts; Support for **group dragging**; Greatly improved the experience of uploading a large number of indicators; Added `swanlab.OpenApi` open interface;
*   2025.05.09: Support for **line chart creation**; Added **data source selection** function to configure chart, supports display of different indicators in single chart; Support for generating **training project GitHub badge**;
*   2025.04.23: Support for line chart **editing**, support for free configuration of X and Y axis data range and title style; Chart search supports **regular expressions**; Support for hardware detection and monitoring of **Kunlunxin XPU**;
*   2025.04.11: Support for line chart **local area selection**; Support for global selection of step range of dashboard line chart; Support for one-click hiding of all charts;
*   2025.04.08: Support for **swanlab.Molecule** data type, support for recording and visualizing biochemical molecular data; Support for saving sorting, filtering, and column order changes in table view;
*   2025.04.07: We have completed joint integration with [EvalScope](https://github.com/ModelScope/EvalScope), and you can now use SwanLab to **evaluate the performance of large models** in EvalScope;
*   2025.03.30: Support for **swanlab.Settings** method, support for more fine-grained experiment behavior control; Support for **Cambricon MLU** hardware monitoring; Support for [Slack notifications](https://docs.swanlab.cn/plugin/notification-slack.html), [Discord notifications](https://docs.swanlab.cn/plugin/notification-discord.html);
*   2025.03.21: 🎉🤗HuggingFace Transformers has officially integrated SwanLab (>=4.50.0 version), [#36433](https://github.com/huggingface/transformers/pull/36433); Added **Object3D charts**, support for recording and visualizing 3D point clouds, [documentation](https://docs.swanlab.cn/api/py-object3d.html); Hardware monitoring supports GPU video memory (MB), disk utilization, network up and down;
*   2025.03.12: 🎉🎉SwanLab **private deployment version** has been released! ! [🔗Deployment documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab now supports plugin extension, such as [email notification](https://docs.swanlab.cn/plugin/notification-email.html), [Feishu notification](https://docs.swanlab.cn/plugin/notification-lark.html)
*   2025.03.09: Support for **experiment sidebar widening**; Added external display Git code button; Added **sync_mlflow** function to support synchronizing experiment tracking with the mlflow framework;
*   2025.03.06: We have completed joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio), and you can now use SwanLab in DiffSynth Studio to **track and visualize the image/video experiments generated by the Diffusion model**, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html);
*   2025.03.04: Added **MLFlow conversion** function, supports converting MLFlow experiments to SwanLab experiments, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html);
*   2025.03.01: Added **move experiment** function, you can now move the experiment to different projects of different organizations;
*   2025.02.24: We have completed joint integration with [EasyR1](https://github.com/hiyouga/EasyR1), and you can now use SwanLab in EasyR1 to **track and visualize multi-modal large model reinforcement learning experiments**, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
*   2025.02.18: We have completed joint integration with [Swift](https://github.com/modelscope/ms-swift), and you can now use SwanLab in Swift's CLI/WebUI to **track and visualize large model fine-tuning experiments**, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html).
*   2025.02.16: Added **chart move grouping, create grouping** function.
*   2025.02.09: We have completed joint integration with [veRL](https://github.com/volcengine/verl), and you can now use SwanLab in veRL to **track and visualize large model reinforcement learning experiments**, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html).
*   2025.02.05: `swanlab.log` supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), adapting to the characteristics of the Jax framework; Support for `name` and `notes` parameters;
*   2025.01.22: Added `sync_tensorboardX` and `sync_tensorboard_torch` functions to support synchronization of experimental tracking with these two TensorBoard frameworks;
*   2025.01.17: Added `sync_wandb` function, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html), supports synchronization of experimental tracking with Weights & Biases; Greatly improved the log rendering performance
*   2025.01.11: The cloud version greatly optimizes the performance of the project table, and supports drag and drop, sorting, filtering and other interactions
*   2025.01.01: Added line chart **persistent smoothing**, line chart drag-and-drop size change, and optimized chart browsing experience
*   2024.12.22: We have completed joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), and you can now use SwanLab in LLaMA Factory to **track and visualize large model fine-tuning experiments**, [usage guide](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger).
*   2024.12.15: **Hardware Monitoring (0.4.0)** function is online, supporting CPU, NPU (Ascend), and GPU (Nvidia) system-level information recording and monitoring.
*   2024.12.06: Added integration of [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html), [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html); Improved the limitation on the length of a single line of log recording.
*   2024.11.26: The hardware section of the environment tab supports recognizing **Huawei Ascend NPU** and **Kunpeng CPU**; The cloud vendor section supports recognizing Qingyun **Jishi Zhisuan**.

</details>

<br>

## 📃 Online Demo

Explore interactive demos to see SwanLab in action:

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![ResNet50 Cat/Dog Classification][demo-cats-dogs-image]][demo-cats-dogs] | [![Yolov8-COCO128 Object Detection][demo-yolo-image]][demo-yolo] |
| Track a simple ResNet50 model trained on a cat/dog image classification task. | Track training hyperparameters and metrics using Yolov8 on the COCO128 dataset. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![Qwen2 Instruction Fine-tuning][demo-qwen2-sft-image]][demo-qwen2-sft] | [![LSTM Google Stock Prediction][demo-google-stock-image]][demo-google-stock] |
| Track instruction fine-tuning of the Qwen2 large language model, completing simple instruction following. | Using a simple LSTM model trained on the Google stock price dataset to predict future stock prices. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![ResNeXt101 Audio Classification][demo-audio-classification-image]][demo-audio-classification] | [![Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiments from ResNet to ResNeXt on audio classification tasks | Lora fine-tuning based on Qwen2-VL multi-modal large models on the COCO2014 dataset. |

| [EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl-image]][demo-easyr1-rl] | [![Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Using the EasyR1 framework for multi-modal LLM RL training | GRPO training based on the Qwen2.5-0.5B model on the GSM8k dataset |

[More examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

If you'd like to experience the latest features, install from source:

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Installation</summary>

[Offline Dashboard documentation](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login & Get API Key

1.  [Register a free account](https://swanlab.cn)
2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and type:

```bash
swanlab login
```

Enter your API Key when prompted and press Enter to complete login.

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

Congratulations!  Head to [SwanLab](https://swanlab.cn) to view your first experiment.

<br>

## 💻 Self-Hosting

The self-hosted community version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy the self-hosted version using Docker

See the documentation for details:  [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For fast installation in China:

```bash
./install.sh
```

To install by pulling the image from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point experiments to the self-hosted service

Log in to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Once logged in, your experiments will be recorded to the self-hosted service.

<br>

## 🔥 Real-World Examples

**Open-source projects using SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): Tutorial on the principles and practice of large language models.
*   [self-llm](https://github.com/datawhalechina/self-llm):  A guide tailored for the Chinese community on quickly fine-tuning (full parameters/Lora) and deploying open-source large language models (LLM) / multi-modal large models (MLLM) in a Linux environment.
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, expansion and reproduction.
*   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL):  Splices the visual head of SmolVLM2 with the Qwen3-0.6B model and fine-tunes it.
*   [OPPO/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models): End-to-end agent foundation models through multi-agent distillation and Agent RL.

**Papers that mention SwanLab:**

*   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
*   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
*   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
*   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorial articles:**

*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cart Pole Inverted Pendulum](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multi-modal Large Model Fine-tuning Practice](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multi-modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA and SwanLab Visualization Recording](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
*   [How to do RAG? SwanLab document assistant solution is open source](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟 If you have tutorials to include, welcome to submit a PR!

<br>

## 🎮 Hardware Monitoring

SwanLab monitors the hardware information and resource usage during AI training. Here's a table showing the supported hardware:

| Hardware       | Information Recording | Resource Monitoring | Script                                                                                                                                |
| :------------- | :-------------------: | :-----------------: | :------------------------------------------------------------------------------------------------------------------------------------ |
| Nvidia GPU     |          ✅           |         ✅          | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)                            |
| Ascend NPU     |          ✅           |         ✅          | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)                            |
| Apple SOC      |          ✅           |         ✅          | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)                              |
| Cambricon MLU  |          ✅           |         ✅          | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py)                      |
| Kunlunxin XPU  |          ✅           |         ✅          | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py)                      |
| Moore Threads GPU |          ✅           |         ✅          | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py)                      |
| MX GPU |          ✅           |         ✅          | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py)                      |
| Hygon DCU |          ✅           |         ✅          | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py)                      |
| CPU            |          ✅           |         ✅          | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)                                      |
| Memory         |          ✅           |         ✅          | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)                                |
| Disk           |          ✅           |         ✅          | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)                                    |
| Network        |          ✅           |         ✅          | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)                              |

Submit an issue or PR if you'd like to record other hardware.

<br>

## 🚗 Framework Integrations

Combine your favorite frameworks with SwanLab! Below are the currently integrated frameworks. Please submit an [Issue](https://github.com/swanhubx/swanlab/issues) to request other integrations.

**Basic Frameworks**

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Fine-tuning Frameworks**

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
*   [MLX-LM](https://docs.swanlab.cn/guide_cloud/integration/integration-mlx-lm.html)

**Evaluation Frameworks**

*   [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)

**Computer Vision**

*   [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
*   [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
*   [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
*   [PaddleDetection](https://docs.