<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: Track, Visualize, and Collaborate on Your Deep Learning Experiments

**SwanLab is an open-source, modern deep learning experiment tracking and visualization tool, making it easy to monitor, compare, and share your AI training runs.**

<a href="https://swanlab.cn">🔥SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Documentation</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report an Issue</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Suggest Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> ·  <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks</a>

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

<div align="center">
  <img src="readme_files/swanlab-overview.png" alt="SwanLab Overview" width="100%"/>
</div>

<br/>

**[English](README_EN.md)** / **[日本語](README_JP.md)** / **[Русский](README_RU.md)** / **中文**

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

## Key Features of SwanLab

*   **Experiment Tracking and Visualization**:
    *   **Intuitive UI**: Visually track your experiments with an easy-to-use web interface.
    *   **Rich Data Types**: Support for scalars, images, audio, text, videos, 3D point clouds, biochemical molecules, and custom ECharts charts.
    *   **Interactive Charts**:  Line charts, media displays, 3D visualizations, and more.

    <div align="center">
    <img src="readme_files/molecule.gif" alt="Molecule Visualization" width="70%"/>
    </div>

    <div align="center">
        <img src="readme_files/echarts.png" alt="ECharts example" width="70%">
    </div>

    *   **LLM-Focused Visualization**:  Text content visualization designed for LLM training, with Markdown rendering support.

    <div align="center">
        <img src="readme_files/text-chart.gif" alt="Text Visualization" width="70%">
    </div>

    *   **Automatic Logging**:  Automatically logs metrics, hyperparameters, hardware resources, Git information, Python environment, and more.
    *   **Resume Support**: Allows for resuming training and adding data to existing experiments.

*   **Framework Integrations**: Seamlessly integrates with **30+** popular frameworks, including:

    <div align="center">
        <img src="readme_files/integrations.png" alt="Framework Integrations" width="70%">
    </div>

*   **Hardware Monitoring**: Real-time monitoring of CPU, GPU (Nvidia, Ascend, etc.), MLU (Cambricon), XPU (Kunlunxin, Metax), DCU (Hygon), memory, disk, and network.

*   **Experiment Management**: Centralized dashboard to manage projects and experiments efficiently.

*   **Result Comparison**:  Compare hyperparameters and results across experiments with tables and charts.

    <div align="center">
        <img src="readme_files/swanlab-table.png" alt="Experiment Table" width="70%">
    </div>

*   **Collaboration**:  Supports collaborative training, enabling team members to view and contribute to experiments in real-time.

*   **Shareable Results**: Easily share experiments with persistent URLs.

*   **Self-Hosting**:  Supports self-hosting for offline use and data privacy. See the [self-hosting guide](#-自托管).

*   **Plugin Extensibility**: Extensible with plugins for [Slack notifications](https://docs.swanlab.cn/plugin/notification-slack.html), [CSV writers](https://docs.swanlab.cn/plugin/writer-csv.html), and more.

> \[!IMPORTANT]
>
> **Star the project** to receive all release notifications directly on GitHub! ⭐️

<div align="center">
<img src="readme_files/star-us.png" alt="Star Us" width="30%"/>
</div>

<br>

## Online Demos

Explore live demonstrations of SwanLab's capabilities:

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![Cats & Dogs Example][demo-cats-dogs-image]][demo-cats-dogs] | [![YOLO Example][demo-yolo-image]][demo-yolo] |
| Tracks a ResNet50 model on a cat/dog image classification task. | Tracks a Yolov8 model during object detection on the COCO128 dataset. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![Qwen2 Example][demo-qwen2-sft-image]][demo-qwen2-sft] | [![LSTM Example][demo-google-stock-image]][demo-google-stock] |
| Tracks instruction fine-tuning of the Qwen2 large language model. |  Uses a simple LSTM model for Google stock price prediction. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Finetuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![Audio Classification Example][demo-audio-classification-image]][demo-audio-classification] | [![Qwen2-VL example][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiments on the audio classification task using ResNet to ResNeXt models | Finetuning Qwen2-VL on COCO2014 dataset using Lora. |

| [EasyR1 Multi-Modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![EasyR1 LLM RL Example][demo-easyr1-rl-image]][demo-easyr1-rl] | [![Qwen2.5-0.5B GRPO example][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Multi-modal LLM RL training using the EasyR1 framework | GRPO training with the Qwen2.5-0.5B model on GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

If you want to try the latest features, you can install from source.

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

1.  [Register](https://swanlab.cn) for a free account.
2.  Log in and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  In your terminal:

```bash
swanlab login
```

Enter your API Key when prompted, then press Enter.

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

Now, head to [SwanLab](https://swanlab.cn) to see your first experiment!

<br>

## Self-Hosting

The self-hosted community version allows you to view the SwanLab dashboard offline.

<div align="center">
  <img src="./readme_files/swanlab-docker.png" alt="Self-Hosting Example" width="70%"/>
</div>

### 1. Deploy a Self-Hosted Version Using Docker

See [the documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html) for details.

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For a quick install in China:

```bash
./install.sh
```

To install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point Experiments to Your Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Once logged in, your experiments will be recorded on your self-hosted service.

<br>

## Real-World Examples

Check out these projects that use SwanLab:

**Open Source Projects Using SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): Large Language Model Tutorial. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm):  Tutorial for Fine-tuning and deploying open-source large language models. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): Understanding and reproducing DeepSeek series work. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
*   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): Finetuning model SmVL2. ![GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)
*   [OPPO/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models): End-to-end foundation model based on multi-agent distillation and agent RL. ![GitHub Repo stars](https://img.shields.io/github/stars/OPPO-PersonalAI/Agent_Foundation_Models)

**Research Papers Using SwanLab:**

*   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
*   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
*   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
*   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorial Articles:**

*   [MNIST Handwriting Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Finetuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Finetuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multimodal Model Finetuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO LLM Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA and SwanLab Visualization](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
*   [How to do RAG? Open-source SwanLab document assistant](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟 We welcome your tutorials! Submit a PR!

<br>

## Hardware Monitoring

SwanLab monitors hardware information and resource usage during AI training:

| Hardware | Information Recording | Resource Monitoring | Script |
| --- | --- | --- | --- |
| NVIDIA GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU | ✅ | ✅ | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC | ✅ | ✅ | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU | ✅ | ✅ | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅ | ✅ | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU     | ✅        | ✅        | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory  | ✅        | ✅        | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk    | ✅        | ✅        | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Open a PR or Issue if you'd like to add support for other hardware!

<br>

## Framework Integrations

Integrate SwanLab with your favorite frameworks!  
Here's a list of currently supported frameworks.  Please submit an [Issue](https://github.com/swanhubx/swanlab/issues) to request integration for a framework.

**Core Frameworks**
- [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
- [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
- [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Finetuning Frameworks**
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

## Plugins and API

Extend SwanLab with plugins to enhance your experiment management!

-   [Customize Your Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
-   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [WeCom Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [File Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open APIs:
-   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparison with Similar Tools

### Tensorboard vs. SwanLab

*   **☁️ Online Support**:
    SwanLab lets you easily sync and save training experiments in the cloud, which makes it simple to check progress remotely, manage your historical projects, share links to your experiments, send instant messages, and look at experiments on multiple devices. Tensorboard is an offline experiment tracking tool.
*   **👥 Collaboration**:
    When doing multi-person, cross-team machine learning collaborations, SwanLab makes it easy to manage multiple people's training projects, share experiment links, and facilitate communication. Tensorboard is primarily designed for individual use and is difficult to use for experiment sharing and collaboration.
*   **💻 Persistent, Centralized Dashboard**:
    Your results are recorded on the same centralized dashboard whether you're training your model on a local computer, in a lab cluster, or on a public cloud GPU instance.
    However, using TensorBoard requires you to spend time copying and managing TFEvent files from various machines.
*   **💪 More Powerful Tables**:
    Using SwanLab tables, you can view, search, and filter results from different experiments. It's easy to view thousands of model versions and discover the best-performing model for various tasks.
    TensorBoard is not suitable for large projects.

### Weights and Biases vs. SwanLab

*   Weights and Biases is a closed-source, MLOps platform that requires an internet connection.
*   SwanLab supports online use but also offers open-source, free, and self-hosted versions.

<br>

## Community

### Related Repositories

*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting deployment scripts.
*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation.
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard):  Offline dashboard (web code).

### Community and Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): For errors and issues.
*   [Email Support](zeyi.lin@swanhub.co): For feedback on SwanLab.
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discussing SwanLab usage and AI techniques.

### SwanLab README Badges

Add SwanLab badges to your README:

[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Your Publications

If SwanLab has been helpful in your research, please consider citing us using the following format:

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

Interested in contributing? Please start by reading the [Contribution Guide](CONTRIBUTING.md).

We welcome support via social media, events, and conferences - thank you!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## License

This repository is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)