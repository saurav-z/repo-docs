<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
    <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
  </picture>

  ## 🚀 SwanLab: Supercharge Your Machine Learning Experiments!

  A powerful, open-source, and modern tool for tracking, visualizing, and collaborating on deep learning experiments. Easily integrate with 30+ frameworks and deploy locally or in the cloud.  Check out the original repo: [https://github.com/SwanHubX/SwanLab](https://github.com/SwanHubX/SwanLab)

  [🔥 SwanLab Online](https://swanlab.cn) · [📃 Documentation](https://docs.swanlab.cn) · [🐞 Report an Issue](https://github.com/swanhubx/swanlab/issues) · [💡 Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) · [📝 Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) · [💬 Community](https://github.com/SwanHubX/assets/blob/main/community.svg) · [🏆 Benchmarks](https://swanlab.cn/benchmarks)

  [![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
  [![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
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

  <img src="readme_files/swanlab-overview.png">

  <br>
  [中文](README_CN.md) / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

  👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

  <a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank">
      <img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ"
           alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" />
  </a>
</div>

<br>

## Table of Contents

*   [🌟 Recent Updates](#-recent-updates)
*   [👋 What is SwanLab?](#-what-is-swanlab)
*   [📃 Online Demo](#-online-demo)
*   [🏁 Quick Start](#-quick-start)
*   [💻 Self-Hosting](#-self-hosting)
*   [🔥 Real-World Examples](#-real-world-examples)
*   [🎮 Hardware Monitoring](#-hardware-monitoring)
*   [🚗 Framework Integrations](#-framework-integrations)
*   [🔌 Plugins & API](#-plugins--api)
*   [🆚 Comparisons with Similar Tools](#-comparisons-with-similar-tools)
*   [👥 Community](#-community)
*   [📃 License](#-license)
*   [⭐ Star History](#-star-history)

<br>

## 🌟 Recent Updates

*   **[Insert latest update highlights here, focusing on new features and improvements. Keep it concise.  Aim for 3-5 bullet points.]**

    *   **[Date]:** [Feature or Enhancement] - [Brief Description].  (e.g., 2024.08.19: Improved chart rendering and low-intrusion loading animation; Integrated MLX-LM and SpecForge.)

    *   **[Date]:** [Feature or Enhancement] - [Brief Description]. (e.g., 2024.08.06: Added training collaboration features (invite collaborators, share project links).)

    *   **[Date]:** [Feature or Enhancement] - [Brief Description]. (e.g., 2024.07.29: Added experiment filtering and sorting in the sidebar.)

*   **[Consider linking to a full changelog for more detailed information]**

<details><summary>Full Update Log</summary>
  *   [Paste the original full changelog here]
</details>

<br>

## 👋 What is SwanLab?

SwanLab is an open-source, lightweight, and modern tool designed for tracking, visualizing, and collaborating on your machine learning experiments. It provides a user-friendly interface and Python API to streamline your model training workflow.

Key Features:

*   **📊 Experiment Tracking & Visualization:** Visualize metrics, hyperparameters, and model outputs.
*   **⚙️ Automated Logging:**  Automatically logs hyperparameters, metrics, hardware information, and more.
*   **💻 Cloud & Offline Support:** Use SwanLab in the cloud or self-host for complete control.
*   **🤝 Team Collaboration:** Share and discuss experiments with your team in real-time.
*   **🔌 Extensive Integrations:** Seamless integration with 30+ popular machine learning frameworks (PyTorch, TensorFlow, etc.).
*   **🎮 Hardware Monitoring:** Real-time monitoring of CPU, GPU, memory, and other system resources.
*   **📦 Flexible Data Types:** Supports scalars, images, audio, text, videos, 3D point clouds, and custom ECharts charts.
*   **🆚 Experiment Comparison:** Compare and analyze different experiments to identify the best models and hyperparameters.
*   **✉️ Share Results:** Share experiment results with persistent URLs, perfect for sharing with colleagues or embedding in reports.
*   **🔌 Plugin Extensions:** Extend SwanLab's functionality with plugins for notifications, data writing, and more.

<br>

## 📃 Online Demo

Explore SwanLab's capabilities with these interactive demos:

| Example                                         | Description                                                                                                      | Link                                                                                       |
| :---------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| [ResNet50 Cat/Dog Classification]                 | Image classification with a simple ResNet50 model on a cat and dog dataset.                                            |  [![][demo-cats-dogs-image]][demo-cats-dogs]    |
| [Yolov8-COCO128 Object Detection]              | Object detection using Yolov8 on the COCO128 dataset, tracks training hyperparameters and metrics.                        |  [![][demo-yolo-image]][demo-yolo]                  |
| [Qwen2 Instruction Fine-tuning]        | Instruction fine-tuning of Qwen2 LLM model.     | [![][demo-qwen2-sft-image]][demo-qwen2-sft]             |
| [LSTM Google Stock Prediction]               | Time series prediction with a LSTM model. |   [![][demo-google-stock-image]][demo-google-stock]            |
| [ResNeXt101 Audio Classification]               | Audio classification using ResNeXt model. |   [![][demo-audio-classification-image]][demo-audio-classification]            |
| [Qwen2-VL COCO Dataset Fine-tuning]               | Fine-tuning Qwen2-VL on COCO dataset. |   [![][demo-qwen2-vl-image]][demo-qwen2-vl]            |
| [EasyR1 LLM RL Training] | LLM RL training with EasyR1.     |  [![][demo-easyr1-rl-image]][demo-easyr1-rl]          |
| [Qwen2.5-0.5B GRPO Training] | GRPO training for Qwen2.5-0.5B model on GSM8k dataset.     |  [![][demo-qwen2-grpo-image]][demo-qwen2-grpo]          |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Code Installation</summary>
Install from source for the latest features:

```bash
git clone https://github.com/SwanHubX/SwanLab.git
cd SwanLab
pip install -e .

# OR
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Extension Installation</summary>
Install offline dashboard support. [Dashboard Docs](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get Your API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Login to your account and get the API key in User Settings -> [API Key](https://swanlab.cn/settings).
3.  Open terminal and run:

```bash
swanlab login
```

Enter your API key when prompted.

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

Your first SwanLab experiment is now live! Visit [SwanLab](https://swanlab.cn) to view it.

<br>

## 💻 Self-Hosting

Self-host SwanLab to maintain data control, or use it in offline environments.

![Swanlab Docker](readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted with Docker

For instructions, see the [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

China installation:

```bash
./install.sh
```

DockerHub installation:

```bash
./install-dockerhub.sh
```

### 2. Direct Experiments to Self-Hosted

Login to your self-hosted instance:

```bash
swanlab login --host http://localhost:8000
```

Log your experiments to your self-hosted service.

<br>

## 🔥 Real-World Examples

Here are some example projects that use SwanLab:
* [Happy-LLM](https://github.com/datawhalechina/happy-llm)
* [Self-LLM](https://github.com/datawhalechina/self-llm)
* [Unlock-Deepseek](https://github.com/datawhalechina/unlock-deepseek)
* [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL)

See the following papers and tutorials that use SwanLab:
* [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
* [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
* [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
* [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

Tutorials:
*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/lstm_stock_prediction.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Finetuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multimodal Fine-tuning Tutorial](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Language Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multimodal Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied AI Tutorial](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA with SwanLab Visualization](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
*   [How to build RAG with SwanLab](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟  If you'd like your tutorial to be included, feel free to submit a PR!

<br>

## 🎮 Hardware Monitoring

SwanLab monitors hardware information:

| Hardware       | Information  | Resource Monitoring | Script                                                            |
| :------------- | :----------- | :------------------ | :---------------------------------------------------------------- |
| NVIDIA GPU     | ✅          | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)   |
| Ascend NPU     | ✅          | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)   |
| Apple SOC      | ✅          | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)   |
| Cambricon MLU  | ✅          | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU  | ✅          | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads  | ✅          | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU      | ✅          | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py)  |
| Hygon DCU      | ✅          | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py)    |
| CPU            | ✅          | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)            |
| Memory         | ✅          | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)         |
| Disk           | ✅          | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)           |
| Network        | ✅          | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)    |

Report any hardware you would like to monitor with an Issue or PR!

<br>

## 🚗 Framework Integrations

Seamlessly integrate SwanLab with your favorite frameworks!  Submit a [Issue](https://github.com/swanhubx/swanlab/issues) for a new framework integration.

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

<br>

## 🔌 Plugins & API

Extend SwanLab's functionality with Plugins!

*   [Customize Your Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [FileLogDir Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparisons with Similar Tools

### TensorBoard vs. SwanLab

*   **☁️ Online Support:** SwanLab provides online experiment synchronization, convenient remote access, historical project management, sharing links, real-time notifications, and multi-device access.  TensorBoard is primarily an offline tool.

*   **👥 Collaboration:** SwanLab excels at managing and sharing training projects with teams, facilitating experiment links, and promoting communication. TensorBoard is designed for individual use, making collaboration difficult.

*   **💻 Persistent, Centralized Dashboard:** View your results in a centralized dashboard, regardless of where your models are trained – local machine, lab cluster, or cloud.  TensorBoard requires manual management of TFEvent files.

*   **💪 Powerful Tables:** SwanLab's tables allow for easy viewing, searching, and filtering across different experiment results, enabling simple comparison and identification of best-performing models. TensorBoard is not optimized for large projects.

### Weights & Biases vs. SwanLab

*   Weights and Biases is a closed-source MLOps platform.

*   SwanLab is open-source, free, and offers self-hosting capabilities.

<br>

## 👥 Community

### Around the Project

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard repository.
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting scripts.

### Community & Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report bugs or ask questions.
*   [Email Support](zeyi.lin@swanhub.co): For any issues or questions.
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discuss SwanLab and learn from the community.

### SwanLab README Badges

Add a SwanLab badge to your README:

[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets at: [assets](https://github.com/SwanHubX/assets)

### Cite SwanLab

If you find SwanLab helpful for your research, please consider citing it:

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

Interested in contributing?  Please read the [Contribution Guide](CONTRIBUTING.md).

Support SwanLab by sharing via social media, events, and conferences!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

<br>

## 📃 License

This project is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)

<!-- link -->

[release-shield]: https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square
[release-link]: https://github.com/swanhubx/swanlab/releases

[license-shield]: https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square
[license-shield-link]: https://github.com/SwanHubX/SwanLab/blob/main/LICENSE

[last-commit-shield]: https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square
[last-commit-shield-link]: https://github.com/swanhubx/swanlab/commits/main

[pypi-version-shield]: https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square
[pypi-version-shield-link]: https://pypi.org/project/swanlab/

[pypi-downloads-shield]: https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square
[pypi-downloads-shield-link]: https://pepy.tech/project/swanlab

[swanlab-cloud-shield]: https://img.shields.io/badge/Product-SwanLab云端版-636a3f?labelColor=black&style=flat-square
[swanlab-cloud-shield-link]: https://swanlab.cn/

[wechat-shield]: https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square
[wechat-shield-link]: https://docs.swanlab.cn/guide_cloud/community/online-support.html

[colab-shield]: https://colab.research.google.com/assets/colab-badge.svg
[colab-shield-link]: https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing

[github-stars-shield]: https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47
[github-stars-link]: https://github.com/swanhubx/swanlab

[github-issues-shield]: https://img.shields.io/github/issues/swanhub