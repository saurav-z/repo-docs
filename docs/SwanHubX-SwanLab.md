<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: Open-Source Deep Learning Experiment Tracking and Visualization Tool

SwanLab empowers AI researchers with an open-source, user-friendly platform to track, visualize, and collaborate on their deep learning experiments, seamlessly integrating with 30+ popular frameworks. [Explore SwanLab on GitHub](https://github.com/SwanHubX/SwanLab).

<p align="center">
    <a href="https://swanlab.cn">🔥 SwanLab Online</a> |
    <a href="https://docs.swanlab.cn">📃 Documentation</a> |
    <a href="https://github.com/swanhubx/swanlab/issues">🐛 Report Issues</a> |
    <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">💡 Feedback</a> |
    <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">✨ Changelog</a> |
    <a href="https://swanlab.cn/benchmarks">🏆 Benchmarks</a>
    <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html">Community</a>
</p>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

<p align="center">
  <img src="readme_files/swanlab-overview.png" alt="SwanLab Overview" width="800">
</p>

<p align="center">
    <a href="README_EN.md">English</a> /
    <a href="README_JP.md">日本語</a> /
    <a href="README_RU.md">Русский</a>
</p>

<p align="center">
    👋 Join our <a href="https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html">WeChat Group</a>
</p>

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking and Visualization:**  Track key metrics, visualize training progress, and gain insights into your experiments with intuitive charts and dashboards.
*   **Comprehensive Framework Integrations:** Seamlessly integrate with over 30 popular machine learning frameworks, including PyTorch, Hugging Face Transformers, and more.
*   **Hardware Monitoring:** Monitor CPU, GPU, NPU (Ascend), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (Muxi), Moore Threads GPU, memory, disk, and network usage.
*   **Experiment Management:** Organize and compare your experiments with a centralized dashboard for easy access and analysis.
*   **Collaboration Features:** Collaborate with your team, share experiment results, and facilitate discussions within your organization.
*   **Self-Hosting Support:** Use SwanLab in both online and offline environments.  Self-hosted version provides the same functionality, allowing you to manage experiments.
*   **Extensibility:** Extend SwanLab's functionality with plugins for notifications, data logging, and more.

<br/>

## Table of Contents

-   [🌟 Recent Updates](#-recent-updates)
-   [👋🏻 What is SwanLab?](#-what-is-swanlab)
-   [📃 Online Demo](#-online-demo)
-   [🏁 Quick Start](#-quick-start)
-   [💻 Self-Hosting](#-self-hosting)
-   [🔥 Real-world Examples](#-real-world-examples)
-   [🎮 Hardware Monitoring](#-hardware-monitoring)
-   [🚗 Framework Integrations](#-framework-integrations)
-   [🔌 Plugins & APIs](#-plugins--apis)
-   [🆚 Comparison with Other Tools](#-comparison-with-other-tools)
-   [👥 Community](#-community)
-   [📃 License](#-license)
<br/>

## 🌟 Recent Updates

-   **2025.07.17:**  📊 More powerful **line chart configuration**, support flexible configuration of line type, color, thickness, grid, legend position, etc.; 📹 Support **swanlab.Video** data type, support recording and visualizing GIF format files; The global chart dashboard supports configuring the Y-axis and the maximum number of experiments displayed;

-   **2025.07.10:**  📚 More powerful **text view**, support Markdown rendering and direction key switching, can be created by `swanlab.echarts.table` and `swanlab.Text`, [Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)

-   **2025.07.06:** 🚄 Support **resume breakpoint training**; New plugin **file recorder**; Integrated [ray](https://github.com/ray-project/ray) framework, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); Integrated [ROLL](https://github.com/volcengine/ROLL) framework, thanks to [@PanAndy](https://github.com/PanAndy), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

-   **2025.06.27:** 📊 Support **local amplification of small line charts**; Support configuring **smoothness of a single line chart**; Greatly improved the interactive effect after the image chart is amplified;

-   **2025.06.20:** 🤗 Integrated [accelerate](https://github.com/huggingface/accelerate) framework, [PR](https://github.com/huggingface/accelerate/pull/3605), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html), enhance the experience of experiment recording in distributed training;

-   **2025.06.18:** 🐜 Integrated [AREAL](https://github.com/inclusionAI/AReaL) framework, thanks to [@xichengpro](https://github.com/xichengpro), [PR](https://github.com/inclusionAI/AReaL/pull/98), [documentation](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process); 🖱 Support mouse Hover to the sidebar experiment, highlight the corresponding curve; Support cross-group comparison of line charts; Support setting the experiment name clipping rule;

-   **2025.06.11:** 📊 Support **swanlab.echarts.table** data type, support plain text chart display; Support **stretching interaction** of groups to increase the number of charts displayed at the same time; Table view adds **indicator maximum/minimum** options;

-   **2025.06.08:** ♻️ Support local storage of complete experimental log files, upload local log files to the cloud/privatized deployment end through **swanlab sync**; Hardware monitoring supports **Hygon DCU**;

-   **2025.06.01:** 🏸 Support **free drag and drop of charts**; Support **ECharts custom charts**, add more than 20 chart types including bar charts, pie charts, histograms; Hardware monitoring supports **Muxi GPU**; Integrated **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** framework;

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

<br>

## 👋🏻 What is SwanLab?

SwanLab is an open-source tool designed to streamline your machine learning workflow. It helps you track, visualize, and collaborate on your experiments, providing a comprehensive platform for experiment tracking and analysis. With SwanLab, you can easily:

*   Visualize training metrics, hyperparameters, and model outputs.
*   Compare and analyze multiple experiments side-by-side.
*   Collaborate with your team and share results.
*   Leverage advanced logging and hardware monitoring.

<p align="center">
    <img src="readme_files/molecule.gif" alt="swanlab-table" width="600">
</p>

<p align="center">
    <img src="readme_files/echarts.png" alt="swanlab-echarts" width="600">
</p>

<p align="center">
    <img src="readme_files/text-chart.gif" alt="text-chart" width="600">
</p>

<br>

## 📃 Online Demo

Explore interactive demos showcasing SwanLab's capabilities:

| Demo                                   | Description                                                     |
| :------------------------------------- | :-------------------------------------------------------------- |
| [ResNet50 猫狗分类][demo-cats-dogs]  | Track and visualize a simple ResNet50 model for image classification on the Cats vs. Dogs dataset. |
| [Yolov8-COCO128 目标检测][demo-yolo] | Track training hyperparameters and metrics of Yolov8 on the COCO128 dataset. |
| [Qwen2 指令微调][demo-qwen2-sft] |  Track Qwen2 Large Language Model instruction finetuning training for instruction following. |
| [LSTM Google 股票预测][demo-google-stock] |  Track training of LSTM model predicting Google stock prices. |
| [ResNeXt101 音频分类][demo-audio-classification] | Track training results from ResNet to ResNeXt on audio classification tasks |
| [Qwen2-VL COCO数据集微调][demo-qwen2-vl] | Track training using Lora for the Qwen2-VL multi-modal large language model on the COCO2014 dataset. |
| [EasyR1 多模态LLM RL训练][demo-easyr1-rl] | Track multi-modal LLM RL training using the EasyR1 framework |
| [Qwen2.5-0.5B GRPO训练][demo-qwen2-grpo] | Track GRPO training of the Qwen2.5-0.5B model on the GSM8k dataset |
[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

If you want to experience the latest features, you can install from source.

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

1.  [Register for a free account](https://swanlab.cn).
2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and enter:

    ```bash
    swanlab login
    ```

    Enter your API Key when prompted and press Enter to complete the login.

### 3. Integrate SwanLab into Your Code

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

You're all set!  Go to [SwanLab](https://swanlab.cn) to view your first SwanLab experiment.

<br>

## 💻 Self-Hosting

Self-hosting allows you to manage your experiments within your own environment.

<p align="center">
  <img src="./readme_files/swanlab-docker.png" alt="SwanLab Docker" width="600">
</p>

### 1. Deploying the Self-Hosted Version with Docker

For detailed instructions, see the [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Quick install for China region:

```bash
./install.sh
```

Install from DockerHub images:

```bash
./install-dockerhub.sh
```

### 2.  Point Experiments to Your Self-Hosted Service

Login to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, your experiments will be recorded to your self-hosted service.

<br>

## 🔥 Real-world Examples

**Open-source projects using SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): A tutorial on Large Language Model (LLM) principles and practice from scratch ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm): A tutorial for Chinese users to fine-tune (full parameters/Lora) and deploy domestic and international open-source large language models (LLM) / multi-modal large language models (MLLM) in a Linux environment. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, expansion and reproduction. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)

**Papers using SwanLab:**

*   [Animation Needs Attention](https://arxiv.org/abs/2507.03916)

**Tutorial Articles:**

*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat and Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - CartPole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multi-modal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multi-modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)

<br>

## 🎮 Hardware Monitoring

SwanLab automatically records and monitors **hardware information** and **resource usage** during AI training. The table below outlines the supported hardware:

| Hardware           | Information Recording | Resource Monitoring | Script                                                                                |
| :----------------- | :-------------------- | :------------------ | :------------------------------------------------------------------------------------ |
| Nvidia GPU         | ✅                   | ✅                 | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)         |
| Ascend NPU         | ✅                   | ✅                 | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)         |
| Apple SOC          | ✅                   | ✅                 | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)          |
| Cambricon MLU      | ✅                   | ✅                 | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py)     |
| Kunlunxin XPU      | ✅                   | ✅                 | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py)   |
| Moore Threads GPU  | ✅                   | ✅                 | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Muxi GPU      | ✅                   | ✅                 | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU      | ✅                   | ✅                 | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU              | ✅                   | ✅                 | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)                 |
| Memory           | ✅                   | ✅                 | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)             |
| Disk             | ✅                   | ✅                 | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)                |
| Network             | ✅                   | ✅                 | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)                |

If you would like to record other hardware, please submit an Issue or PR!

<br>

## 🚗 Framework Integrations

Use your favorite frameworks with SwanLab!  Here is a list of our integrations, and we welcome [Issues](https://github.com/swanhubx/swanlab/issues) to request integrations.

**Base Frameworks**

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
*   [Easy