<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
    <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
  </picture>

  <h1>SwanLab: Open-Source, Modern Deep Learning Training Tracker and Visualizer</h1>
  <p><b>Effortlessly track, visualize, and collaborate on your deep learning experiments with SwanLab, an open-source tool designed for ease of use and powerful insights.</b></p>

  <a href="https://swanlab.cn">🔥 SwanLab Online</a> | <a href="https://docs.swanlab.cn">📃 Documentation</a> | <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> | <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> | <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> | <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks</a>

  [![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
  [![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
  [![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
  [![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
  [![GitHub Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
  [![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
  [![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
  [![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
  [![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
  [![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
  [![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

  <br/>
  <img src="readme_files/swanlab-overview.png" alt="SwanLab Overview" width="80%"/>
  <br/>

  <a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>

<br/>

## Key Features

*   **Experiment Tracking**: Effortlessly log metrics, hyperparameters, and more with simple Python API calls.
*   **Flexible Visualization**: Interactive dashboards for visualizing training progress, comparing experiments, and identifying trends.
*   **Framework Integrations**: Seamless integration with popular deep learning frameworks like PyTorch, TensorFlow, and others.
*   **Hardware Monitoring**: Real-time monitoring of CPU, GPU, memory, and other hardware resources.
*   **Collaboration**: Share experiments and collaborate with your team using online project pages.
*   **Self-Hosted**: Use SwanLab locally or on your own servers for complete control over your data.
*   **Customization**:  Extensible via plugins for email notifications, custom chart types, and more.

<br/>

## Table of Contents

*   [🌟 Recent Updates](#-recent-updates)
*   [👋🏻 What is SwanLab?](#-what-is-swanlab)
*   [📃 Online Demo](#-online-demo)
*   [🏁 Quick Start](#-quick-start)
*   [💻 Self-Hosting](#-self-hosting)
*   [🔥 Practical Examples](#-practical-examples)
*   [🎮 Hardware Monitoring](#-hardware-monitoring)
*   [🚗 Framework Integrations](#-framework-integrations)
*   [🔌 Plugins and API](#-plugins-and-api)
*   [🆚 Comparison with Similar Tools](#-comparison-with-similar-tools)
*   [👥 Community](#-community)
*   [📃 License](#-license)
*   [Star History](#star-history)

<br/>

## 🌟 Recent Updates

**(Summarized - see original README for full details)**

*   **[2025.08.19]** Improved chart rendering, low-intrusion loading animations. Integration of [MLX-LM](https://github.com/ml-explore/mlx-lm) and [SpecForge](https://github.com/sgl-project/SpecForge).
*   **[2025.08.06]** Training Collaboration feature release and Workspaces new views.
*   **[2025.07.29]** Experiment filtering/sorting and Column control panel.
*   **[2025.07.17]** Enhanced line chart configuration and `swanlab.Video` support for GIF files.
*   **[2025.07.10]** Improved text views with Markdown support and the ability to create tables using `swanlab.echarts.table` and `swanlab.Text`.
*   **[2025.07.06]** Resume training, File Recorder Plugin.
*   **[2025.06.27]** Zoomable line charts and Smooth line charts.
*   **[2025.06.20]** Integration with [Hugging Face Accelerate](https://github.com/huggingface/accelerate)
*   **[2025.06.18]** Integration with [AREAL](https://github.com/inclusionAI/AReaL)
*   **[2025.06.11]** Supports `swanlab.echarts.table` data type. Dragging to increase the display number of charts.
*   **[2025.06.08]** Support local experiments.
*   **[2025.06.01]** Support ECharts custom charts and hardware monitoring for 沐曦GPU.
*   **[2025.05.25]** The standard error stream of logs.
*   **[2025.05.14]** Experiment Tags, Log Scale and Group drag support.

<details><summary>Complete Changelog</summary>

See the original README for full changelog details.
</details>

<br>

## 👋🏻 What is SwanLab?

SwanLab is an open-source, modern, and lightweight tool for tracking and visualizing your deep learning experiments. It provides a comprehensive platform to track, record, compare, and collaborate on your machine learning projects. With a user-friendly Python API and an intuitive UI, SwanLab empowers researchers to:

*   **Visualize Training Progress**: Understand your experiments with interactive charts and visualizations.
*   **Automated Logging**: Effortlessly track metrics, hyperparameters, and system resource usage.
*   **Experiment Comparison**: Compare experiments side-by-side to analyze results and identify optimal configurations.
*   **Collaborate**: Facilitate team collaboration through online experiment sharing and discussions.

https://github.com/user-attachments/assets/7965fec4-c8b0-4956-803d-dbf177b44f54

**Key Features:**

1.  **📊 Experiment Metrics and Hyperparameter Tracking**: Integrate SwanLab into your ML pipeline with simple Python calls to track key metrics.

    *   ☁️ Support Cloud use (similar to Weights & Biases), Check the training progress at any time and any place. [How to check experiments on your phone](https://docs.swanlab.cn/guide_cloud/general/app.html)
    *   📝 Support **hyperparameter recording**, **metric summary**, and **table analysis**.
    *   🌸 **Visualize Training Process**: Visualize the results of each step of the experiment through the UI interface, allowing trainers to intuitively see the results of each step of the experiment, analyze the trend of indicators, and determine which changes have led to the improvement of the model effect, thereby comprehensively improving the efficiency of model iteration.
    *   **Supported Metadata Types**: Scalar metrics, images, audio, text, video, 3D point cloud, biochemical molecules, Echarts custom charts...

    ![swanlab-table](readme_files/molecule.gif)

    *   **Supported Chart Types**: Line charts, media charts (images, audio, text, video), 3D point cloud, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, [custom charts](https://docs.swanlab.cn/guide_cloud/experiment_track/log-custom-chart.html)...

    [![swanlab-echarts](readme_files/echarts.png)](https://swanlab.cn/@ZeyiLin/swanlab-echarts-demo/charts)

    *   **LLM Generated Content Visualization Component**: Text content visualization chart created for large language model training scenarios, supporting Markdown rendering

    ![text-chart](readme_files/text-chart.gif)

    *   **Automatic Background Recording**: Log logging, hardware environment, Git repository, Python environment, Python library list, project running directory
    *   **Resume Training Record**: Supports supplementing new metric data to the same experiment after training is completed/interrupted

2.  ⚡️ **Comprehensive Framework Integration**: Ready to use with PyTorch, 🤗HuggingFace Transformers, PyTorch Lightning, 🦙LLaMA Factory, MMDetection, Ultralytics, PaddleDetetion, LightGBM, XGBoost, Keras, Tensorboard, Weights&Biases, OpenAI, Swift, XTuner, Stable Baseline3, Hydra and more than **30+** frameworks

    ![](readme_files/integrations.png)

3.  💻 **Hardware Monitoring**: Monitor hardware metrics in real-time, including CPU, NPU (**Ascend**), GPU (**Nvidia**), MLU (**Cambricon**), XPU (**Kunlunxin**), DCU (**Hygon**), MetaX GPU (**沐曦XPU**), Moore Threads GPU (**摩尔线程**), and memory usage.

4.  📦 **Experiment Management**: Manage projects and experiments using a centralized dashboard.

5.  🆚 **Compare Results**: Compare hyperparameters and results across experiments using online tables and charts to spark new insights.

    ![](readme_files/swanlab-table.png)

6.  👥 **Online Collaboration**: Collaborate with your team, synchronize experiments in a shared project, view training records, and exchange feedback.

7.  ✉️ **Share Results**: Share experiments with ease by sharing persistent URLs for each experiment.

8.  💻 **Self-Hosting Support**: Use SwanLab offline with a community-based self-hosted version that offers a dashboard and experiment management capabilities.  See [Self-Hosting Instructions](#-self-hosting).

9.  🔌 **Plugin Extensibility**: Expand SwanLab's functionality through plugins such as [Lark notifications](https://docs.swanlab.cn/plugin/notification-lark.html), [Slack notifications](https://docs.swanlab.cn/plugin/notification-slack.html), [CSV recorder](https://docs.swanlab.cn/plugin/writer-csv.html), and more.

>   \[!IMPORTANT]
>   **Star the project** to receive notifications about new releases and updates! ⭐️

![star-us](readme_files/star-us.png)

<br>

## 📃 Online Demo

Explore SwanLab's capabilities with these online demos:

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Tracks a simple ResNet50 model trained on a cat and dog dataset. | Tracks the training of Yolov8 on the COCO128 dataset. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Tracks Qwen2 Large Language Model instruction fine-tuning training to follow instructions. | Train a simple LSTM model on the Google stock price dataset to predict future stock prices. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiments from ResNet to ResNeXt on audio classification tasks. | Lora fine-tuning on the Qwen2-VL multi-modal large model with the COCO2014 dataset. |

| [EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Using the EasyR1 framework for multi-modal LLM RL training | GRPO training on the Qwen2.5-0.5B model with the GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

Install from source if you want to try the latest features.

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

### 2. Login and Get Your API Key

1.  [Register](https://swanlab.cn) for a free account.

2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).

3.  Open your terminal and run:

```bash
swanlab login
```

Enter your API Key when prompted and press Enter to log in.

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

That's it!  Head to [SwanLab](https://swanlab.cn) to view your first experiment.

<br>

## 💻 Self-Hosting

Self-hosting allows you to view your SwanLab dashboard offline.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted Version with Docker

See documentation for details: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Quick Installation for China:

```bash
./install.sh
```

Install by Pulling the Image from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Direct Experiments to the Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, your experiments will be recorded on the self-hosted service.

<br>

## 🔥 Practical Examples

**Open source projects that use SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): Large language model tutorial from zero.  ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm): Tutorial on fine-tuning and deploying open-source large language models (LLMs) and multi-modal large language models (MLLMs). ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series working interpretation and expansion.  ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
*   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): The visual head of SmolVLM2 is spliced with the Qwen3-0.6B model for fine-tuning.  ![GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)

**Excellent papers that use SwanLab:**

*   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
*   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
*   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
*   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorials:**

*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat and Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pre-training](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multi-modal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multi-modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence Tutorial](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA and SwanLab Visualization Recording](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)

🌟 Contribute tutorials or projects!  Submit a PR.

<br>

## 🎮 Hardware Monitoring

SwanLab tracks **hardware information** and **resource usage** during your AI training.  Here's the hardware support:

| Hardware         | Information Recording | Resource Monitoring | Script                                                                    |
| :--------------- | :-------------------- | :------------------ | :------------------------------------------------------------------------ |
| NVIDIA GPU       | ✅                     | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)       |
| Ascend NPU       | ✅                     | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)       |
| Apple SOC        | ✅                     | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)       |
| Cambricon MLU    | ✅                     | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py)     |
| Kunlunxin XPU    | ✅                     | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py)   |
| Moore Threads GPU | ✅                     | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| 沐曦GPU          | ✅                     | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py)         |
| 海光DCU          | ✅                     | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py)          |
| CPU              | ✅                     | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)               |
| Memory           | ✅                     | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)            |
| Disk             | ✅                     | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)              |
| Network          | ✅                     | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)         |

Contribute a hardware monitoring script! Submit an issue or PR.

<br>

## 🚗 Framework Integrations

Easily use SwanLab with your favorite frameworks! Below is the list of frameworks we have integrated.  Submit an [Issue](https://github.com/swanhubx/swanlab/issues) if you want to suggest a new integration.

**Core Frameworks**
- [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
- [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
- [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Fine-tuning Frameworks**
- [PyTorch Lightning](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-lightning.html)
- [HuggingFace Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-transformers.html)
- [LLaMA Factory](https://docs.swanlab.cn/guide_cloud/integration/integration-llama-factory.html)
- [Modelscope Swift](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html)
- [DiffSynth Studio](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html)
- [Sentence Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-sentence-transformers.html)
- [PaddleNLP](https://docs.swanlab.cn/guide_cloud/integration/integration-paddlenlp.html)
- [OpenMind](https://modelers.cn/docs/zh/openmind-library/1.0.0/basic_tutorial/finetune/finetune_pt.html#%E8%AE%AD%E7%BB%83%E7%9B%91%E6%8E%A7)
- [Torchtune](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-torchtune.html)
- [XTuner](https://docs.swanlab.cn/guide_cloud/integration/integration-xtuner.html)
- [MMEngine](https://docs.swanlab.cn/guide_cloud/integration/integration-mmengine.html)
- [FastAI](https://docs.swanlab.cn/guide_cloud/integration/integration-fastai.html)
- [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html)
- [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html)

**Evaluation Frameworks**
- [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)

**Computer Vision**
- [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
- [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
- [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
- [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
- [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)

**Reinforcement Learning**
- [Stable Baseline3](https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html)
- [veRL](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)
- [HuggingFace trl](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-trl.html)
- [EasyR1](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
- [AReaL](https://docs.swanlab.cn/guide_cloud/integration/integration-areal.html)
- [ROLL](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

**Other Frameworks:**
- [Tensorboard](https://docs.swanlab.cn/guide_cloud/integration/integration-tensorboard.html)
- [Weights&Biases](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html)
- [MLFlow](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html)
- [HuggingFace Accelerate](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html)
- [Ray](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html)
- [Unsloth](https://docs.swanlab.cn/guide_cloud/integration/integration-unsloth.html)
- [Hydra](https://docs.swanlab.cn/guide_cloud/integration/integration-hydra.html)
- [Omegaconf](https://docs.swanlab.cn/guide_cloud/integration/integration-omegaconf.html)
- [OpenAI](https://docs.swanlab.cn/guide_cloud/integration/integration-openai.html)
- [ZhipuAI](https://docs.swanlab.cn/guide_cloud/integration/integration-zhipuai.html)

[More Integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br>

## 🔌 Plugins and API

Extend SwanLab's capabilities with plugins.

-   [Customize Your Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
-   [Lark Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [WeCom Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV Recorder](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [File Log Recorder](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:
- [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparison with Similar Tools

### Tensorboard vs SwanLab

-   **☁️ Online Support**: SwanLab seamlessly syncs training experiments online, which benefits remote training, project management, experiment sharing, real-time message notifications, and multi-terminal experiment viewing.  Tensorboard is a local, offline experiment tracking tool.

-   **👥 Collaboration**: SwanLab facilitates multi-user machine learning projects, easy experiment management, and experiment sharing and exchange within and across teams. Tensorboard is generally for individual use.

-   **💻 Persistent and Centralized Dashboard**: Your results are recorded in a centralized dashboard, regardless of where you train your models.  Tensorboard requires more effort for file management and copying.

-   **💪 More Powerful Tables**: SwanLab tables allow you to view, search, and filter results from different experiments, allowing you to easily review thousands of model versions and find the best-performing model for different tasks. TensorBoard is not designed for large-scale projects.

### Weights and Biases vs SwanLab

-   Weights and Biases is a closed-source MLOps platform that requires an internet connection.