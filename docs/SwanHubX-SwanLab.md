<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

**Supercharge your machine learning experiments with SwanLab, the open-source, modern, and user-friendly training tracking and visualization tool.** Seamlessly integrated with 30+ popular frameworks and offering both cloud and offline functionality, SwanLab helps you track, analyze, and collaborate on your AI projects effortlessly.

<a href="https://swanlab.cn">🔥SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Docs</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks</a>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)


![](readme_files/swanlab-overview.png)

[English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>

<br/>

## Key Features

*   **Experiment Tracking & Visualization**: Visualize training metrics, monitor progress, and identify trends through intuitive charts and dashboards.
    *   Track Scalars, Images, Audio, Text, Video, 3D Point Clouds, and Custom Echarts charts.
    *   Comprehensive Chart Types: Line charts, media charts (images, audio, text, video), 3D point clouds, custom charts, and more.

    ![swanlab-table](readme_files/molecule.gif)

    ![swanlab-echarts](readme_files/echarts.png)

    ![text-chart](readme_files/text-chart.gif)

*   **Framework Integrations**: Seamlessly integrate with 30+ popular machine learning frameworks (PyTorch, Hugging Face Transformers, etc.).
*   **Hardware Monitoring**: Real-time tracking of CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (MxS), Moore Threads GPU, memory and network resources.
*   **Experiment Management**: Centralized dashboard for easy organization, comparison, and management of your experiments.
*   **Results Comparison**: Compare hyperparameters and results across experiments with interactive tables and charts to accelerate insights.
    ![swanlab-table](readme_files/swanlab-table.png)
*   **Collaboration**: Enable collaborative training with team members, share experiments, and facilitate discussions for improved team efficiency.
*   **Sharing**: Generate persistent URLs to share experiment results easily with collaborators.
*   **Self-Hosting**: Use SwanLab offline with self-hosted version. [How to](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)
*   **Plugin Extensibility**: Extend SwanLab's functionality with plugins for notifications and more.

> \[!IMPORTANT]
>
> **Star the project** to receive notifications for future releases! ⭐️

![star-us](readme_files/star-us.png)

<br>

## Table of Contents

*   [🚀 Recent Updates](#-recent-updates)
*   [👋🏻 What is SwanLab?](#-what-is-swanlab)
*   [📃 Online Demo](#-online-demo)
*   [🏁 Quickstart](#-quickstart)
*   [💻 Self-Hosting](#-self-hosting)
*   [🔥 Case Studies](#-case-studies)
*   [🎮 Hardware Monitoring](#-hardware-monitoring)
*   [🚗 Framework Integrations](#-framework-integrations)
*   [🔌 Plugins and API](#-plugins-and-api)
*   [🆚 Comparisons](#-comparisons)
*   [👥 Community](#-community)
*   [📃 License](#-license)
*   [Star History](#star-history)

<br/>

## 🚀 Recent Updates

-   **2025.08.19**: Enhanced chart rendering performance and loading animations, focused on experiment analysis; integrated MLX-LM and SpecForge.
-   **2025.08.06**: Training collaboration, workspace list view, experiment tags; column control panel, multi-API Key management, and new charts.
-   **2025.07.29**: Experiment filtering and sorting; table view with column controls; multi-API key management; new PR, ROC, and confusion matrix charts.
-   **2025.07.17**: Improved line chart configuration; swanlab.Video data type for GIF files; Y-axis and max experiment display count in global dashboard.
-   **2025.07.10**: Enhanced text view with Markdown rendering; swanlab.echarts.table and swanlab.Text support, Demo.
-   **2025.07.06**: Resume training support; new file recorder plugin; integration of ray.
-   **2025.06.27**: Zoom in for line charts; smoothed single line charts; interaction improvements for image charts.
-   **2025.06.20**: Integrated accelerate framework, enhancing the distributed training experience.
-   **2025.06.18**: Integrated the AREAL framework, and the feature of highlighting corresponding curves when hovering experiments in the sidebar.

<details><summary>Full Changelog</summary>

-   2025.06.11: Added swanlab.echarts.table data type. Added table options: Index max/min
-   2025.06.08: Added full experiment log files saving and hardware monitor supported for HaiGuang DCU.
-   2025.06.01: Added charts free drag, ECharts custom charts, and hardware monitor supported for MuXi GPU. Integrated PaddleNLP.
-   2025.05.25: Added standard error flow logging, PyTorch Lightning prints can be better logged; and hardware monitoring supported for Moore Threads. Added run command logging security, and API Key is hidden.
-   2025.05.14: Support experiment Tags and Log Scale for line chart. Dragging groups and optimizing the experience of uploading a large number of indicators
-   2025.05.09: Line chart creation and added the configuration of data sources for the chart. Support GitHub badges for training projects.
-   2025.04.23: Support line chart editing. Support the range and title of the chart. Chart search supports regular expressions. Support Kunlunxin XPU hardware detection and monitoring.
-   2025.04.11: Support selecting local area selection in line charts. Support a one-click hide of all the charts.
-   2025.04.08: Supports swanlab.Molecule data type to record and visualize biochemical molecular data. Added states to the table, e.g. sorting, filtering, column changes
-   2025.04.07: We have completed the joint integration with [EvalScope](https://github.com/ModelScope/EvalScope), now you can use SwanLab to assess large model performance in EvalScope.
-   2025.03.30: Supports swanlab.Settings method and supports fine-grained control of experiments, MLU hardware monitoring. Supports Slack and Discord notifications.
-   2025.03.21: HuggingFace Transformers has officially integrated SwanLab(>=4.50.0), New Object3D chart, which supports the recording and visualization of 3D point clouds; GPU memory(MB), disk utilization, and network up and down are also recorded.
-   2025.03.12: 🎉🎉SwanLab private deployment has been released! [🔗Deployment documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab now supports plugin extensions, such as [email notifications](https://docs.swanlab.cn/plugin/notification-email.html) and [Feishu notifications](https://docs.swanlab.cn/plugin/notification-lark.html).
-   2025.03.09: Support expanding experiment sidebar; New display Git code button and add the sync_mlflow function.
-   2025.03.06: We have completed the joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio), now you can use SwanLab to track and visualize diffusion model experiments in DiffSynth Studio.
-   2025.03.04: Added the MLFlow conversion function, which supports converting MLFlow experiments to SwanLab experiments.
-   2025.03.01: Added the function of moving experiments.
-   2025.02.24: We have completed the joint integration with [EasyR1](https://github.com/hiyouga/EasyR1), now you can use SwanLab to track and visualize multimodal large model reinforcement learning experiments in EasyR1.
-   2025.02.18: We have completed the joint integration with [Swift](https://github.com/modelscope/ms-swift), now you can use SwanLab to track and visualize large model fine-tuning experiments in Swift CLI/WebUI.
-   2025.02.16: Added the function of chart moving and creating groups.
-   2025.02.09: We have completed the joint integration with [veRL](https://github.com/volcengine/verl), now you can use SwanLab to track and visualize large model reinforcement learning experiments in veRL.
-   2025.02.05: `swanlab.log` supports nested dictionaries, adaptive Jax framework features, supports the `name` and `notes` parameters.
-   2025.01.22: Added the `sync_tensorboardX` and `sync_tensorboard_torch` functions, which supports synchronizing experiment tracking with these two TensorBoard frameworks.
-   2025.01.17: Added the `sync_wandb` function, and significantly improved the performance of log rendering.
-   2025.01.11: Cloud version has significantly optimized the performance of the project table and supports dragging, sorting, and filtering.
-   2025.01.01: Added persistent smoothing for line charts, dragging the line chart to change the size and optimize the chart browsing experience.
-   2024.12.22: We have completed the joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), now you can use SwanLab to track and visualize large model fine-tuning experiments in LLaMA Factory.
-   2024.12.15: Hardware monitoring (0.4.0) function is available, which supports the system-level information recording and monitoring of CPU, NPU (Ascend), and GPU (Nvidia).
-   2024.12.06: Added integration with [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html) and [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html); Increased the limit on the length of a single line of log records.
-   2024.11.26: Environment tab - hardware part supports the recognition of Huawei Ascend NPU and Kunpeng CPU; Cloud vendor part supports the recognition of Qingyun Foundation Intelligence.
</details>

<br>

## 📃 Online Demo

Explore SwanLab's capabilities with these interactive demos:

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![Cats & Dogs][demo-cats-dogs-image]][demo-cats-dogs] | [![YOLOv8][demo-yolo-image]][demo-yolo] |
| Tracks a simple ResNet50 model trained on a cat/dog image classification task. | Uses Yolov8 for object detection on the COCO128 dataset, tracking training hyperparameters and metrics. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![Qwen2 SFT][demo-qwen2-sft-image]][demo-qwen2-sft] | [![Google Stock][demo-google-stock-image]][demo-google-stock] |
| Tracks Qwen2 Large Language Model instruction fine-tuning. | Trains a simple LSTM model on Google stock price data. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![Audio Classification][demo-audio-classification-image]][demo-audio-classification] | [![Qwen2-VL COCO][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Demonstrates the progress from ResNet to ResNeXt in audio classification. | Fine-tuning the Qwen2-VL multimodal model on the COCO2014 dataset with LoRA. |

| [EasyR1 Multi-modal LLM RL][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![EasyR1 RL][demo-easyr1-rl-image]][demo-easyr1-rl] | [![Qwen2.5 GRPO][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| RL training with the EasyR1 framework. | GRPO training of Qwen2.5-0.5B model on the GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

For the latest features, install from source.

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Board Extension Installation</summary>

[Offline Board Documentation](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Log in and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  In your terminal:

```bash
swanlab login
```

Enter your API Key when prompted, and press Enter to log in.

### 3. Integrate SwanLab into your Code

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

Visit [SwanLab](https://swanlab.cn) to view your first experiment.

<br>

## 💻 Self-Hosting

The self-hosted community version allows you to view the SwanLab dashboard offline.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy a Self-Hosted Version with Docker

Refer to the [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Quick installation for China mainland users:

```bash
./install.sh
```

Install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Direct Experiments to Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Once logged in, experiment records will be saved to your self-hosted service.

<br>

## 🔥 Case Studies

**Open-source projects using SwanLab:**

-   [happy-llm](https://github.com/datawhalechina/happy-llm)
-   [self-llm](https://github.com/datawhalechina/self-llm)
-   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek)
-   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL)

**Papers using SwanLab:**

-   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
-   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
-   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
-   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorials:**
-   [MNIST Handwritten Digits Recognition](https://docs.swanlab.cn/examples/mnist.html)
-   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
-   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
-   [Resnet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
-   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
-   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
-   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
-   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
-   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
-   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
-   [Stable Diffusion Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
-   [LLM Pre-training](https://docs.swanlab.cn/examples/pretrain_llm.html)
-   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
-   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
-   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
-   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
-   [Qwen2-VL Multimodal Large Model Fine-tuning Tutorial](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
-   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
-   [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
-   [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
-   [GLM-4.5-Air-LoRA and SwanLab Visualization Records](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)

🌟 PRs for any other tutorials would be appreciated!

<br>

## 🎮 Hardware Monitoring

SwanLab records **hardware information** and **resource usage** during AI training:

| Hardware | Information Recording | Resource Monitoring | Script |
| :------- | :-------------------- | :------------------ | :----- |
| NVIDIA GPU | ✅                   | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU | ✅                   | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC  | ✅                   | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU | ✅                   | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅                   | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅                   | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU | ✅                   | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅                   | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU | ✅ | ✅ | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory | ✅ | ✅ | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk | ✅ | ✅ | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Submit issues and PRs for other hardware!

<br>

## 🚗 Framework Integrations

Integrate your favorite frameworks with SwanLab!

Here's a list of the frameworks we've integrated.  Please submit an [Issue](https://github.com/swanhubx/swanlab/issues) to request framework integrations.

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

Enhance your experiment management with SwanLab plugins!

*   [Create Your Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeCom Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Directory Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparisons

### Tensor