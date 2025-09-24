<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

## SwanLab: Supercharge Your Deep Learning Experiments

**SwanLab is an open-source, user-friendly tool that empowers you to track, visualize, and collaborate on your deep learning experiments with ease.**

<a href="https://swanlab.cn">🔥SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Documentation</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report an Issue</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks Community</a>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

![](readme_files/swanlab-overview.png)

[中文 / English / 日本語 / Русский](README_EN.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>

<br/>

## Table of Contents

- [🌟 Recent Updates](#-recent-updates)
- [💡 What is SwanLab?](#-what-is-swanlab)
- [🚀 Online Demo](#-online-demo)
- [🏁 Quickstart](#-quickstart)
- [🐳 Self-Hosting](#-self-hosting)
- [🔥 Real-World Examples](#-real-world-examples)
- [⚙️ Hardware Monitoring](#-hardware-monitoring)
- [🧩 Framework Integrations](#-framework-integrations)
- [🔌 Plugins & APIs](#-plugins--apis)
- [🆚 Comparison with Existing Tools](#-comparison-with-existing-tools)
- [🤝 Community](#-community)
- [📄 License](#-license)
- [⭐ Star History](#-star-history)

<br/>

## 🌟 Recent Updates

*   **[Insert Updates Here - Summarize and Highlight Key Features from the original update log - Consider grouping them by feature type or date]**
    *   2025.09.22: 📊 New UI released; table view supports global sorting and filtering; data layer unified table view and chart view;
    *   2025.09.12: 🔢 Support for creating **scalar charts**, flexibly display statistical values ​​of experimental indicators; major upgrade of the organization management page, providing more powerful permission control and project management capabilities;
    *   ...
<details><summary>Full Changelog</summary>
    *   2025.06.20: 🤗 Integrated the [accelerate](https://github.com/huggingface/accelerate) framework, [PR](https://github.com/huggingface/accelerate/pull/3605), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html), enhancing the experimental recording experience in distributed training;

    *   2025.06.18: 🐜 Integrated the [AREAL](https://github.com/inclusionAI/AReaL) framework, thanks to [@xichengpro](https://github.com/xichengpro), [PR](https://github.com/inclusionAI/AReaL/pull/98), [documentation](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process); 🖱 Supports highlighting the corresponding curves when the mouse hovers over the experimental sidebar; Supports cross-group comparison of line charts; Supports setting experiment name cropping rules;

    *   2025.06.11: 📊 Support **swanlab.echarts.table** data type, support pure text chart display; support for **stretching and interacting** with groups to increase the number of charts displayed at the same time; the table view adds **index maximum/minimum** options;

    *   2025.06.08: ♻️ Support storing complete experimental log files locally, and upload local log files to the cloud/privately deployed end through **swanlab sync**; hardware monitoring supports **Hygon DCU**;

    *   2025.06.01: 🏸 Support **chart free dragging**; support **ECharts custom charts**, adding 20+ chart types including bar charts, pie charts, and histograms; hardware monitoring supports **Muxi GPU**; integrates the **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** framework;

    *   2025.05.25: The log supports recording **standard error streams**, and the print information of PyTorch Lightning and other frameworks can be better recorded; hardware monitoring supports **Moore Threads**; added a security protection function for recording running commands, and the API Key will be automatically hidden;

    *   2025.05.14: Supports **experimental tags**; supports line chart **Log Scale**; supports **group dragging**; greatly optimizes the experience of uploading a large number of indicators; adds the `swanlab.OpenApi` open interface;

    *   2025.05.09: Supports **line chart creation**; the configuration chart function adds the **data source selection** function, and supports displaying different indicators in a single chart; supports generating **training project GitHub badges**;

    *   2025.04.23: Supports line chart **editing**, supports freely configuring the X and Y axis data ranges and title styles of the chart; the chart search supports **regular expressions**; supports hardware detection and monitoring of **Kunlunxin XPU**;

    *   2025.04.11: Supports line chart **local area selection**; supports global selection of the step range of the dashboard line chart; supports one-click hiding of all charts;

    *   2025.04.08: Supports the **swanlab.Molecule** data type, supports recording and visualizing biochemical molecular data; supports saving the sorting, filtering, and column order change status in the table view;

    *   2025.04.07: We have completed the joint integration with [EvalScope](https://github.com/ModelScope/EvalScope). Now you can use SwanLab in EvalScope to **evaluate large model performance**;

    *   2025.03.30: Supports the **swanlab.Settings** method, supports more refined experimental behavior control; supports **Cambricon MLU** hardware monitoring; supports [Slack notification](https://docs.swanlab.cn/plugin/notification-slack.html), [Discord notification](https://docs.swanlab.cn/plugin/notification-discord.html);

    *   2025.03.21: 🎉🤗HuggingFace Transformers has officially integrated SwanLab (>=4.50.0 version), [#36433](https://github.com/huggingface/transformers/pull/36433); added the **Object3D chart**, supports recording and visualizing 3D point clouds, [documentation](https://docs.swanlab.cn/api/py-object3d.html); hardware monitoring supports GPU video memory (MB), disk utilization, and network uplink and downlink records;

    *   2025.03.12: 🎉🎉SwanLab **private deployment version** has been released! ! [🔗Deployment documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab has supported plugin extensions, such as [email notifications](https://docs.swanlab.cn/plugin/notification-email.html), [Feishu notifications](https://docs.swanlab.cn/plugin/notification-lark.html)

    *   2025.03.09: Supports **experiment sidebar widening**; new external display Git code button; new **sync_mlflow** function, supports syncing experiment tracking with the mlflow framework;

    *   2025.03.06: We have completed the joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio). Now you can use SwanLab in DiffSynth Studio to **track and visualize Diffusion model text-to-image/video experiments**, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html);

    *   2025.03.04: Added the **MLFlow conversion** function, supports converting MLFlow experiments to SwanLab experiments, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html);

    *   2025.03.01: Added the **move experiment** function, and now you can move the experiment to different projects in different organizations;

    *   2025.02.24: We have completed the joint integration with [EasyR1](https://github.com/hiyouga/EasyR1), now you can use SwanLab in EasyR1 to **track and visualize multi-modal large model reinforcement learning experiments**, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)

    *   2025.02.18: We have completed the joint integration with [Swift](https://github.com/modelscope/ms-swift), now you can use SwanLab in Swift's CLI/WebUI to **track and visualize large model fine-tuning experiments**, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html).

    *   2025.02.16: Added the function of **moving and grouping charts, creating groups**.

    *   2025.02.09: We have completed the joint integration with [veRL](https://github.com/volcengine/verl), now you can use SwanLab in veRL to **track and visualize large model reinforcement learning experiments**, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html).

    *   2025.02.05: `swanlab.log` supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), adapts to Jax framework characteristics; supports the `name` and `notes` parameters;

    *   2025.01.22: Added the functions `sync_tensorboardX` and `sync_tensorboard_torch`, supports synchronizing experimental tracking with these two TensorBoard frameworks;

    *   2025.01.17: Added the `sync_wandb` function, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html), supports syncing experiment tracking with Weights & Biases; greatly improved the log rendering performance

    *   2025.01.11: The cloud version has greatly optimized the performance of the project table and supports drag and drop, sorting, filtering and other interactions

    *   2025.01.01: Added line chart **persistent smoothing**, and drag-and-drop line chart resizing to optimize the chart browsing experience

    *   2024.12.22: We have completed the joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), and now you can use SwanLab in LLaMA Factory to **track and visualize large model fine-tuning experiments**, [Usage Guide](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger).

    *   2024.12.15: **Hardware monitoring (0.4.0)** function is launched, supporting the recording and monitoring of system-level information for CPU, NPU (Ascend), and GPU (Nvidia).

    *   2024.12.06: Added integration with [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html) and [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html); improved the limit on the length of a single line of log records.
</details>

<br/>

## 💡 What is SwanLab?

SwanLab is an open-source, lightweight tool designed for tracking and visualizing machine learning experiments. It provides a comprehensive platform to monitor, record, compare, and collaborate on your ML projects.  With SwanLab, researchers can easily visualize training progress, log metrics, manage hyperparameters, compare experiments, and collaborate with teams, all through an intuitive interface and easy-to-use Python API.

Key Features:

*   **Experiment Tracking and Visualization:** Monitor training metrics, visualize data, and analyze model performance with interactive charts and dashboards.
*   **Automated Logging:** Easily log metrics, hyperparameters, system information, and more with minimal code integration.
*   **Framework Compatibility:** Seamlessly integrates with over 30 popular ML frameworks, including PyTorch, TensorFlow, and Hugging Face Transformers.
*   **Experiment Comparison:** Compare multiple experiments side-by-side to identify trends, optimize hyperparameters, and gain valuable insights.
*   **Collaboration Features:** Share experiments, invite collaborators, and foster a collaborative environment for research and development.
*   **Self-Hosting Option:** Deploy SwanLab on your own infrastructure for complete data control and privacy.
*   **Extensible with Plugins:** Customize your SwanLab experience with plugins for notifications, data writing, and more.

<br/>

## 🚀 Online Demo

Explore live demonstrations of SwanLab's capabilities:

| [ResNet50 Cats & Dogs Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![Cats & Dogs][demo-cats-dogs-image]][demo-cats-dogs] | [![YOLOv8][demo-yolo-image]][demo-yolo] |
| Track a simple ResNet50 model image classification task trained on the Cats & Dogs dataset. |  Use Yolov8 for object detection on COCO128, and track training metrics and hyperparameters. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![Qwen2 Instruction Fine-tuning][demo-qwen2-sft-image]][demo-qwen2-sft] | [![Google Stock Prediction][demo-google-stock-image]][demo-google-stock] |
| Track the instruction fine-tuning of a Qwen2 LLM, demonstrating instruction following. | Train a simple LSTM model on Google stock prices to predict future prices. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![ResNeXt101 Audio Classification][demo-audio-classification-image]][demo-audio-classification] | [![Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiments, from ResNet to ResNeXt, for audio classification tasks. | Fine-tuning on Qwen2-VL, a multi-modal large model, utilizing LoRA on the COCO2014 dataset. |

| [EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl-image]][demo-easyr1-rl] | [![Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Multi-modal LLM RL training using the EasyR1 framework | GRPO training on the GSM8k dataset, based on the Qwen2.5-0.5B model |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br/>

## 🏁 Quickstart

Get started with SwanLab in a few easy steps:

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

If you want to experience the latest features, you can install from source.

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Extension</summary>

[Offline Dashboard Documentation](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get API Key

1.  [Register for a free account](https://swanlab.cn).
2.  Log in and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and run:

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

That's it! View your first SwanLab experiment at [SwanLab](https://swanlab.cn).

<br/>

## 🐳 Self-Hosting

Self-hosting allows you to run SwanLab on your own infrastructure.

![SwanLab Docker](./readme_files/swanlab-docker.png)

### 1. Deploying a Self-Hosted Version with Docker

For detailed instructions, please refer to the documentation: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

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

Login to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, your experiments will be recorded on your self-hosted service.

<br/>

## 🔥 Real-World Examples

Explore these projects using SwanLab:

**Tutorials & Open-Source Projects:**
-   [happy-llm](https://github.com/datawhalechina/happy-llm): LLM Tutorial from scratch, [GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
-   [self-llm](https://github.com/datawhalechina/self-llm): An instruction manual for fine-tuning LLMs and MLLMs, [GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
-   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): A DeepSeek series of work interpretation, expansion and replication, [GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
-   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): SmolVLM2 visual header with Qwen3-0.6B fine-tuning, [GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)
-   [OPPO/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models): End-to-end agent foundation models via multi-agent distillation and Agent RL. [GitHub Repo stars](https://img.shields.io/github/stars/OPPO-PersonalAI/Agent_Foundation_Models)

**Research Papers:**
-   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
-   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
-   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
-   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorials:**
-   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
-   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
-   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
-   [Resnet Cats & Dogs Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
-   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
-   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
-   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
-   [DQN Reinforcement Learning - CartPole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
-   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
-   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
-   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
-   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
-   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
-   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
-   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
-   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
-   [Qwen2-VL Multi-modal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
-   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
-   [Qwen3-SmVL-0.6B Multi-modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
-   [LeRobot Embodied Intelligence Guide](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
-   [GLM-4.5-Air-LoRA & SwanLab Visualization](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
-   [How to do RAG? SwanLab Document Assistant Program is Open Source](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟  Submit a pull request to include your tutorial!

<br/>

## ⚙️ Hardware Monitoring

SwanLab provides real-time hardware monitoring during AI training:

| Hardware        | Information Recorded | Resource Monitoring | Script                                                              |
| --------------- | -------------------- | ------------------- | ------------------------------------------------------------------- |
| NVIDIA GPU      | ✅                   | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)      |
| Ascend NPU      | ✅                   | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)      |
| Apple SOC       | ✅                   | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)      |
| Cambricon MLU   | ✅                   | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU   | ✅                   | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅                   | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU      | ✅                   | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU       | ✅                   | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU             | ✅                   | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)        |
| Memory          | ✅                   | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)   |
| Disk            | ✅                   | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)      |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Submit an issue or PR if you need additional hardware support!

<br/>

## 🧩 Framework Integrations

Integrate SwanLab with your favorite frameworks!

We support a wide range of ML frameworks:

**Base Frameworks**
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
- [MLX-LM](https://docs.swanlab.cn/guide_cloud/integration/integration-mlx-lm.html)

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