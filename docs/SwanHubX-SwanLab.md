<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## SwanLab: Open-Source AI Experiment Tracking and Visualization

**SwanLab** is an open-source, modern tool designed to streamline your deep learning training process by providing powerful tracking, visualization, and collaboration features; [check out the original repo here](https://github.com/SwanHubX/SwanLab).

<div align="center">
  
  [🔥 SwanLab Online](https://swanlab.cn) | [📃 Documentation](https://docs.swanlab.cn) | [🐛 Report Issues](https://github.com/swanhubx/swanlab/issues) | [💡 Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) | [🚀 Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) |  <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> [基线社区](https://swanlab.cn/benchmarks)
  
</div>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)


<div align="center">
  <img src="readme_files/swanlab-overview.png" alt="SwanLab Overview" width="80%">
</div>

**Languages:** 中文 / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Effortless Integration:** Seamlessly integrate SwanLab into your existing machine learning projects with Python API support for various frameworks.
*   **Comprehensive Tracking:** Automatically log metrics, hyperparameters, hardware information, and more, to visualize your training process.
*   **Rich Visualization:** Explore your experiments with intuitive charts, graphs, and interactive dashboards.
*   **Flexible Deployment:** Use it locally, on the cloud (similar to Weights & Biases), or deploy it on your own servers.
*   **Collaboration:** Share your experiments easily and collaborate with team members.
*   **Framework Support:** Compatible with 30+ popular machine learning frameworks.
*   **Hardware Monitoring:** Real-time monitoring and recording of hardware resources (CPU, GPU, memory, etc.).
*   **Extensible:** Add new functionalities with flexible plugin system.

## Table of Contents

*   [🌟 Recent Updates](#-最近更新)
*   [👋🏻 What is SwanLab?](#-什么是swanlab)
*   [📃 Online Demo](#-在线演示)
*   [🏁 Quickstart](#-快速开始)
*   [💻 Self-Hosting](#-自托管)
*   [🔥 Real-World Examples](#-实战案例)
*   [🎮 Hardware Monitoring](#-硬件记录)
*   [🚗 Framework Integrations](#-框架集成)
*   [🔌 Plugins and API](#-插件与api)
*   [🆚 Comparisons](#-与熟悉的工具的比较)
*   [👥 Community](#-社区)
*   [📃 License](#-协议)
*   [⭐ Star History](#-star-history)

<br/>

## 🌟 Recent Updates

(Updates listed - keep it concise and focused on key enhancements.  Focus on last few months)

*   **2025.07.17:** Enhanced line chart configuration with more options, swanlab.Video data type to support GIF format, global chart dashboard Y axis settings.
*   **2025.07.10:** Improved text view with Markdown rendering and arrow key navigation.
*   **2025.07.06:** Support for resume training; New file logger plugin; Integration with Ray and ROLL frameworks.
*   **2025.06.27:** Zoom in local zoom for line charts; Support for single line chart smoothing; Improved zoom interaction for image charts.
*   **2025.06.20:** Integration with the accelerate framework, enhancing the experience of experiment record during distributed training.

(Remaining updates can be removed.  The point is to show some recent updates but not overwhelm the reader.  Keep the README concise)

<details><summary>Complete Changelog</summary>

-   2025.05.25：日志支持记录**标准错误流**，PyTorch Lightning等框架的打印信息可以被更好地记录；硬件监控支持**摩尔线程**；新增运行命令记录安全防护功能，API Key将被自动隐藏；

-   2025.05.14：支持**实验Tag**；支持折线图**Log Scale**；支持**分组拖拽**；大幅度优化了大量指标上传的体验；增加`swanlab.OpenApi`开放接口；

-   2025.05.09：支持**折线图创建**；配置图表功能增加**数据源选择**功能，支持单张图表显示不同的指标；支持生成**训练项目GitHub徽章**；

-   2025.04.23：支持折线图**编辑**，支持自由配置图表的X、Y轴数据范围和标题样式；图表搜索支持**正则表达式**；支持**昆仑芯XPU**的硬件检测与监控；

-   2025.04.11：支持折线图**局部区域选取**；支持全局选择仪表盘折线图的step范围；支持一键隐藏全部图表；

-   2025.04.08：支持**swanlab.Molecule**数据类型，支持记录与可视化生物化学分子数据；支持保存表格视图中的排序、筛选、列顺序变化状态；

-   2025.04.07：我们与 [EvalScope](https://github.com/ModelScope/EvalScope) 完成了联合集成，现在你可以在EvalScope中使用SwanLab来**评估大模型性能**；

-   2025.03.30：支持**swanlab.Settings**方法，支持更精细化的实验行为控制；支持**寒武纪MLU**硬件监控；支持 [Slack通知](https://docs.swanlab.cn/plugin/notification-slack.html)、[Discord通知](https://docs.swanlab.cn/plugin/notification-discord.html)；

-   2025.03.21：🎉🤗HuggingFace Transformers已正式集成SwanLab（>=4.50.0版本），[#36433](https://github.com/huggingface/transformers/pull/36433)；新增 **Object3D图表** ，支持记录与可视化三维点云，[文档](https://docs.swanlab.cn/api/py-object3d.html)；硬件监控支持了 GPU显存（MB）、磁盘利用率、网络上下行 的记录；

-   2025.03.12：🎉🎉SwanLab**私有化部署版**现已发布！！[🔗部署文档](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)；SwanLab 已支持插件扩展，如 [邮件通知](https://docs.swanlab.cn/plugin/notification-email.html)、[飞书通知](https://docs.swanlab.cn/plugin/notification-lark.html)

-   2025.03.09：支持**实验侧边栏拉宽**；新增外显 Git代码 按钮；新增 **sync_mlflow** 功能，支持与mlflow框架同步实验跟踪；

-   2025.03.06：我们与 [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio) 完成了联合集成，现在你可以在DiffSynth Studio中使用SwanLab来**跟踪和可视化Diffusion模型文生图/视频实验**，[使用指引](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html)；

-   2025.03.04：新增 **MLFlow转换** 功能，支持将MLFlow实验转换为SwanLab实验，[使用指引](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html)；

-   2025.03.01：新增 **移动实验** 功能，现在可以将实验移动到不同组织的不同项目下了；

-   2025.02.24：我们与 [EasyR1](https://github.com/hiyouga/EasyR1) 完成了联合集成，现在你可以在EasyR1中使用SwanLab来**跟踪和可视化多模态大模型强化学习实验**，[使用指引](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)

-   2025.02.18：我们与 [Swift](https://github.com/modelscope/ms-swift) 完成了联合集成，现在你可以在Swift的CLI/WebUI中使用SwanLab来**跟踪和可视化大模型微调实验**，[使用指引](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html)。

-   2025.02.16：新增 **图表移动分组、创建分组** 功能。

-   2025.02.09：我们与 [veRL](https://github.com/volcengine/verl) 完成了联合集成，现在你可以在veRL中使用SwanLab来**跟踪和可视化大模型强化学习实验**，[使用指引](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)。

-   2025.02.05：`swanlab.log`支持嵌套字典 [#812](https://github.com/SwanHubX/SwanLab/pull/812)，适配Jax框架特性；支持`name`与`notes`参数；

-   2025.01.22：新增`sync_tensorboardX`与`sync_tensorboard_torch`功能，支持与此两种TensorBoard框架同步实验跟踪；

-   2025.01.17：新增`sync_wandb`功能，[文档](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html)，支持与Weights & Biases实验跟踪同步；大幅改进了日志渲染性能

-   2025.01.11：云端版大幅优化了项目表格的性能，并支持拖拽、排序、筛选等交互

-   2025.01.01：新增折线图**持久化平滑**、折线图拖拽式改变大小，优化图表浏览体验

-   2024.12.22：我们与 [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 完成了联合集成，现在你可以在LLaMA Factory中使用SwanLab来**跟踪和可视化大模型微调实验**，[使用指引](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger)。

-   2024.12.15：**硬件监控（0.4.0）** 功能上线，支持CPU、NPU（Ascend）、GPU（Nvidia）的系统级信息记录与监控。

-   2024.12.06：新增对[LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html)、[XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html)的集成；提高了对日志记录单行长度的限制。

-   2024.11.26：环境选项卡-硬件部分支持识别**华为昇腾NPU**与**鲲鹏CPU**；云厂商部分支持识别青云**基石智算**。

</details>

<br/>

## 👋🏻 What is SwanLab?

SwanLab is an open-source experiment tracking and visualization tool that helps you monitor, analyze, and share your machine learning experiments efficiently.  It offers a user-friendly experience with a Python API and a clean UI to track, record, compare, and collaborate on your experiments. It includes: training visualization, automatic log recording, hyperparameter recording, experimental comparison, and multi-person collaboration.

<div align="center">
  <img src="readme_files/swanlab-table.png" alt="SwanLab Table Example" width="60%">
</div>

### Key Features Summarized

*   **Experiment Tracking & Visualization:** Track metrics, visualize training progress with insightful charts, and analyze trends.

*   **Comprehensive Logging:** Automatically log hyperparameters, metrics, and system information.

*   **Framework Integration:** Easily integrates with popular frameworks such as PyTorch, Hugging Face Transformers, LightGBM, XGBoost and more.

*   **Hardware Monitoring:** Monitor real-time hardware usage (GPU, CPU, memory, etc.).

*   **Experiment Management:** Manage projects and experiments through an easy-to-use dashboard.

*   **Comparison & Analysis:** Compare and contrast different experiments to gain insights and improve model iterations.

*   **Collaboration:** Collaborate on experiments with your team.

*   **Sharing:** Share experiments easily via shareable URLs.

*   **Self-Hosting:** Supports offline use and self-hosting for local or private deployments.

*   **Plugin Ecosystem:**  Enhance SwanLab with plugins.

> \[!IMPORTANT]
>
> **Star the project** to receive release notifications without delay! ⭐️

<div align="center">
  <img src="readme_files/star-us.png" alt="Star Us" width="25%">
</div>

<br/>

## 📃 Online Demo

See SwanLab in action with these interactive demos:

| Demo                                      | Description                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| [ResNet50 Cats vs. Dogs Classification]   | Tracking a ResNet50 model for image classification on the Cats vs. Dogs dataset. |
| [Yolov8-COCO128 Object Detection]         | Tracking of a YOLOv8 model for object detection on the COCO128 dataset.      |
| [Qwen2 Instruction Fine-tuning]          | Instructions fine-tuning the Qwen2 large language models.                        |
| [LSTM Google Stock Prediction]            | Predicting Google stock prices using a simple LSTM model.                       |
| [ResNeXt101 Audio Classification] | Progressive experiments using ResNet to ResNeXt for audio classification tasks |
| [Qwen2-VL COCO Dataset Fine-tuning] | Lora fine-tuning based on the Qwen2-VL multi-modal large language model and COCO2014 dataset. |
| [EasyR1 Multi-modal LLM RL Training] | RL training using EasyR1 framework for multi-modal LLM tasks. |
| [Qwen2.5-0.5B GRPO Training] | GRPO training based on Qwen2.5-0.5B model using GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

[demo-cats-dogs]: https://swanlab.cn/@ZeyiLin/Cats_Dogs_Classification/runs/jzo93k112f15pmx14vtxf/chart
[demo-yolo]: https://swanlab.cn/@ZeyiLin/ultratest/runs/yux7vclmsmmsar9ear7u5/chart
[demo-qwen2-sft]: https://swanlab.cn/@ZeyiLin/Qwen2-fintune/runs/cfg5f8dzkp6vouxzaxlx6/chart
[demo-google-stock]:https://swanlab.cn/@ZeyiLin/Google-Stock-Prediction/charts
[demo-audio-classification]:https://swanlab.cn/@ZeyiLin/PyTorch_Audio_Classification/charts
[demo-qwen2-vl]:https://swanlab.cn/@ZeyiLin/Qwen2-VL-finetune/runs/pkgest5xhdn3ukpdy6kv5/chart
[demo-easyr1-rl]:https://swanlab.cn/@Kedreamix/easy_r1/runs/wzezd8q36bb6dlza6wtpc/chart
[demo-qwen2-grpo]:https://swanlab.cn/@kmno4/Qwen-R1/runs/t0zr3ak5r7188mjbjgdsc/chart

<br/>

## 🏁 Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

Install from source code if you'd like to try the latest features.

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Installation</summary>

[Offline Dashboard Documentation](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Log in to your account, and go to User Settings -> [API Key](https://swanlab.cn/settings) to copy your API Key.
3.  Open your terminal and enter:

```bash
swanlab login
```

When prompted, enter your API Key and press Enter.

### 3. Integrate SwanLab into Your Code

```python
import swanlab

# Initialize a new SwanLab experiment
swanlab.init(
    project="my-first-ml",
    config={'learning-rate': 0.003},
)

# Record metrics
for i in range(10):
    swanlab.log({"loss": i, "acc": i})
```

Congratulations! Visit [SwanLab](https://swanlab.cn) to view your first experiment.

<br/>

## 💻 Self-Hosting

The self-hosted community version allows for offline viewing of the SwanLab dashboard.

<div align="center">
  <img src="./readme_files/swanlab-docker.png" alt="SwanLab Docker" width="40%">
</div>

### 1. Deploy Self-Hosted with Docker

See the documentation for details:  [Docs](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Install quickly for China:

```bash
./install.sh
```

Install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Specify Experiment to Self-Hosted Service

Login to self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After login, the experiment can be recorded in self-hosted service.

<br/>

## 🔥 Real-World Examples

*   [happy-llm](https://github.com/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek)
*   [Animation Needs Attention](https://arxiv.org/abs/2507.03916)

(Also link to tutorial articles, but keep it concise.)

<br/>

## 🎮 Hardware Monitoring

SwanLab monitors hardware information and resource usage during AI training. Supported hardware includes:

| Hardware         | Information Recording | Resource Monitoring | Script                                                                          |
| ---------------- | --------------------- | ------------------- | ------------------------------------------------------------------------------- |
| NVIDIA GPU       | ✅                   | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)         |
| Ascend NPU       | ✅                   | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)       |
| Apple SOC        | ✅                   | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)         |
| Cambricon MLU    | ✅                   | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py)      |
| Kunlunxin XPU    | ✅                   | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py)   |
| Moore Threads GPU | ✅                   | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU        | ✅                   | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py)         |
| Hygon DCU        | ✅                   | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py)         |
| CPU              | ✅                   | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)             |
| Memory           | ✅                   | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)        |
| Disk             | ✅                   | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)           |
| Network          | ✅                   | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)       |

Contribute to recording other hardware!

<br/>

## 🚗 Framework Integrations

Use SwanLab with your favorite frameworks! Below is a list of frameworks we have integrated, and you are welcome to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to suggest frameworks you wish to integrate.

**Frameworks**
-   PyTorch
-   MindSpore
-   Keras
-   PyTorch Lightning
-   HuggingFace Transformers
-   LLaMA Factory
-   Modelscope Swift
-   DiffSynth Studio
-   Sentence Transformers
-   PaddleNLP
-   OpenMind
-   Torchtune
-   XTuner
-   MMEngine
-   FastAI
-   LightGBM
-   XGBoost
-   EvalScope
-   Ultralytics
-   MMDetection
-   MMSegmentation
-   PaddleDetection
-   PaddleYOLO
-   Stable Baseline3
-   veRL
-   HuggingFace trl
-   EasyR1
-   AReaL
-   ROLL
-   Tensorboard
-   Weights&Biases
-   MLFlow
-   HuggingFace Accelerate
-   Ray
-   Unsloth
-   Hydra
-   Omegaconf
-   OpenAI
-   ZhipuAI

[More Integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br/>

## 🔌 Plugins and API

Extend the functionality of SwanLab with plugins to enhance your experiment management!

*   [Customize Your Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Logger](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Logger](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI:
-   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br/>

## 🆚 Comparisons

### TensorBoard vs. SwanLab

*   **Cloud Support:** SwanLab provides convenient cloud synchronization and storage of training experiments for easy remote access. TensorBoard is offline.
*   **Collaboration:** SwanLab makes it easier to manage training projects and collaborate, sharing links, and cross-space discussion. TensorBoard is primarily designed for individuals.
*   **Persistent Dashboard:** SwanLab stores results in a centralized dashboard, whether training locally, on a cluster, or in the cloud, while TensorBoard requires effort to copy and manage TFEvent files.
*   **Advanced Table Feature:** SwanLab tables allow for viewing, searching, and filtering results from different experiments. TensorBoard is not suitable for large projects.

### Weights and Biases vs. SwanLab

*   Weights and Biases is a closed-source, cloud-based MLOps platform.
*   SwanLab is open-source, free, and offers self-hosting.

<br/>

## 👥 Community

### Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Documentation Repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline Dashboard Repository
*   [self-hosted](https://github.com/swanhubx/self-hosted): Private Deployment Script Repository

### Community and Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report issues and ask questions.
*   [Email Support](zeyi.lin@swanhub.co): Feedback on using SwanLab.
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discuss issues and share AI technology.

### SwanLab README Badges

Add a SwanLab badge to your README:

[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![SwanLab Visualization](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design resources: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Papers

If SwanLab has been helpful in your research, cite it:

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

Read the [Contributing Guide](CONTRIBUTING.md).

Share SwanLab via social media, events, and conferences!

<br/>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br/>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## 📃 License

This project is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)