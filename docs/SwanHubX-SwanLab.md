<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: Track, Visualize, and Collaborate on Your Deep Learning Experiments

**SwanLab is an open-source, user-friendly tool designed to streamline your deep learning workflow by providing comprehensive experiment tracking, visualization, and collaboration features.**  Effortlessly integrate with 30+ popular ML frameworks and enjoy cloud and offline support.

[![][release-shield]][release-link]
[![][dockerhub-shield]][dockerhub-link]
[![][github-stars-shield]][github-stars-link]
[![][github-issues-shield]][github-issues-shield-link]
[![][github-contributors-shield]][github-contributors-link]
[![][license-shield]][license-shield-link]  
[![][tracking-swanlab-shield]][tracking-swanlab-shield-link]
[![][last-commit-shield]][last-commit-shield-link]
[![][pypi-version-shield]][pypi-version-shield-link]
[![][pypi-downloads-shield]][pypi-downloads-shield-link]
[![][colab-shield]][colab-shield-link]

[🔥 SwanLab Online Version](https://swanlab.cn) · [📃 Documentation](https://docs.swanlab.cn) · [Report Issues](https://github.com/swanhubx/swanlab/issues) · [Suggest Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) · [Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> [Community Benchmarks](https://swanlab.cn/benchmarks)

![](readme_files/swanlab-overview.png)

[中文 / English / 日本語 / Русский](README_EN.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking and Visualization:**  Visualize your training progress with interactive charts, track key metrics, and analyze experiment trends.
    *   Supports Scalar metrics, images, audio, text, video, 3D point clouds, and custom ECharts charts.
    *   Includes line charts, media views, and specialized LLM content visualization.
*   **Comprehensive Framework Integrations:**  Seamlessly integrate with over 30 popular deep learning frameworks, including PyTorch, Hugging Face Transformers, and many more.
*   **Hardware Monitoring:** Real-time monitoring of CPU, GPU (Nvidia, Ascend, and more), memory, disk, and network utilization.
*   **Experiment Management & Comparison:** Organize and compare experiments using a centralized dashboard, facilitating data-driven insights and iterative improvements.
*   **Collaboration:** Enable team-based training with real-time experiment synchronization and sharing.
*   **Shareable Results:** Generate persistent URLs for easy sharing of experiments.
*   **Self-Hosting Support:** Utilize the self-hosted community version for offline experiment tracking.
*   **Plugin & API Extensions:** Expand SwanLab's functionality through customizable plugins and a user-friendly API.

<br>

## Recent Updates

-   **(2025.09.22)**:  📊 New UI released, table views support global sorting and filtering; unified table and chart view for data.
-   **(2025.09.12)**:  🔢 Scalar chart support; enhanced organization and project management features.
-   **(2025.08.19)**:  🤔 Improved chart rendering and low-intrusion loading animation; integrated MLX-LM and SpecForge frameworks.
-   **(2025.08.06)**:  👥 Training collaboration features released, support project co-workers, share project links and QR codes; workspace support for list views.
-   **(2025.07.29)**:  🚀 Sidebar supports experiment filtering and sorting; table view column control panel released.
-   **(2025.07.17)**:  📊 More flexible line chart configuration; Video support for GIF files.

<details><summary>More Recent Updates</summary>

-   **(2025.07.10)**:📚 Stronger Text view, support Markdown rendering and arrow keys; Text created by `swanlab.echarts.table` and `swanlab.Text`.
-   **(2025.07.06)**:🚄 Support resume breakpoints; new plugin, file recorder; integrated Ray and ROLL framework
-   **(2025.06.27)**:📊Support for small line chart zoom; Support for configuring single line chart smoothing.
-   **(2025.06.20)**:🤗 Integrated [accelerate](https://github.com/huggingface/accelerate) framework
-   **(2025.06.18)**:🐜 Integrated [AREAL](https://github.com/inclusionAI/AReaL) framework
-   **(2025.06.11)**:📊 Support for **swanlab.echarts.table** data type; support for **stretching interaction** of groups
-   **(2025.06.08)**:♻️ Local experiments log; hardware monitoring for **Hygon DCU**
-   **(2025.06.01)**:🏸 Support for **chart free drag**; Support for **ECharts custom chart**, with 20+ chart types
-   **(2025.05.25)**: Log support for **standard error stream**
-   **(2025.05.14)**: Support for **experiment Tag**
-   **(2025.05.09)**: Support for **line chart creation**
-   **(2025.04.23)**: Support for line chart **editing**
-   **(2025.04.11)**: Support for line chart **local area selection**
-   **(2025.04.08)**: Support for **swanlab.Molecule** data type
-   **(2025.04.07)**: We've integrated with [EvalScope](https://github.com/ModelScope/EvalScope)
-   **(2025.03.30)**: Support for **swanlab.Settings** method
-   **(2025.03.21)**:🎉🤗HuggingFace Transformers has been officially integrated with SwanLab; New **Object3D chart**
-   **(2025.03.12)**:🎉🎉SwanLab **private deployment version** has been released!
-   **(2025.03.09)**: Support **experiment sidebar widening**; New external Git code button; New **sync_mlflow**
-   **(2025.03.06)**: We've integrated with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio)
-   **(2025.03.04)**: New **MLFlow conversion**
-   **(2025.03.01)**: New **move experiment**
-   **(2025.02.24)**: We've integrated with [EasyR1](https://github.com/hiyouga/EasyR1)
-   **(2025.02.18)**: We've integrated with [Swift](https://github.com/modelscope/ms-swift)
-   **(2025.02.16)**: New **Chart moving grouping and create grouping**
-   **(2025.02.09)**: We've integrated with [veRL](https://github.com/volcengine/verl)
-   **(2025.02.05)**:`swanlab.log` support nested dictionaries
-   **(2025.01.22)**: Added `sync_tensorboardX` and `sync_tensorboard_torch`
-   **(2025.01.17)**: Added `sync_wandb`
-   **(2025.01.11)**: Significantly optimized project table performance
-   **(2025.01.01)**: New line chart **persistent smoothing**
-   **(2024.12.22)**: We've integrated with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
-   **(2024.12.15)**: **Hardware monitoring (0.4.0)** function online
-   **(2024.12.06)**: Added integration of [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html) and [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html)
-   **(2024.11.26)**: Hardware section of the environment tab supports recognition of **Huawei Ascend NPU** and **Kunpeng CPU**

</details>

<br>

## Online Demos

| [ResNet50 猫狗分类][demo-cats-dogs] | [Yolov8-COCO128 目标检测][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| 跟踪一个简单的 ResNet50 模型在猫狗数据集上训练的图像分类任务。 | 使用 Yolov8 在 COCO128 数据集上进行目标检测任务，跟踪训练超参数和指标。 |

| [Qwen2 指令微调][demo-qwen2-sft] | [LSTM Google 股票预测][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| 跟踪 Qwen2 大语言模型的指令微调训练，完成简单的指令遵循。 | 使用简单的 LSTM 模型在 Google 股价数据集上训练，实现对未来股价的预测。 |

| [ResNeXt101 音频分类][demo-audio-classification] | [Qwen2-VL COCO数据集微调][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| 从ResNet到ResNeXt在音频分类任务上的渐进式实验过程 | 基于Qwen2-VL多模态大模型，在COCO2014数据集上进行Lora微调。 |

| [EasyR1 多模态LLM RL训练][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO训练][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| 使用EasyR1框架进行多模态LLM RL训练 | 基于Qwen2.5-0.5B模型在GSM8k数据集上进行GRPO训练 |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Install offline board extension</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Log in and get your API Key

1.  [Register](https://swanlab.cn) for free.
2.  Log in and copy your API Key from [API Key](https://swanlab.cn/settings).
3.  Open your terminal:

```bash
swanlab login
```

Enter your API Key when prompted.

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

Your first SwanLab experiment is ready!  Go to [SwanLab](https://swanlab.cn) to view.

<br>

## Self-Hosting

Self-hosted community version supports offline viewing of SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy the self-hosted version using Docker

For details, please refer to: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Chinese region fast install:

```bash
./install.sh
```

Install from DockerHub image:

```bash
./install-dockerhub.sh
```

### 2. Direct experiments to self-hosted services

Log in to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, you can record experiments to the self-hosted service.

<br>

## Practical Examples

**Tutorials & Open Source Projects Using SwanLab:**
- [happy-llm](https://github.com/datawhalechina/happy-llm): Large language model principle and practice tutorials from scratch ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
- [self-llm](https://github.com/datawhalechina/self-llm): "Open Source Large Model Guide"
- [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek Series work interpretation, extension and reproduction ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
- [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): Fine-tuning the visual head of SmolVLM2 with the Qwen3-0.6B model. ![GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)
- [OPPO/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models): End-to-end Agent foundation model through multi-Agent distillation and Agent RL. ![GitHub Repo stars](https://img.shields.io/github/stars/OPPO-PersonalAI/Agent_Foundation_Models)

**Papers Using SwanLab:**
- [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
- [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
- [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
- [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorials:**
- [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
- [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
- [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
- [Resnet Cat and Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
- [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
- [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
- [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
- [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
- [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
- [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
- [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
- [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
- [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
- [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
- [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
- [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
- [Qwen2-VL Multi-modal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
- [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
- [Qwen3-SmVL-0.6B Multi-modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
- [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
- [GLM-4.5-Air-LoRA and SwanLab Visualization Recording](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
- [How to do RAG? Open source solution of SwanLab document assistant is here](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟 If you have a tutorial you want to include, welcome to submit a PR!

<br>

## Hardware Monitoring

SwanLab records **hardware information** and **resource usage** during AI training.  Here's a table of supported hardware:

| Hardware | Information Recording | Resource Monitoring | Script                                                                 |
| :------- | :-------------------- | :------------------ | :--------------------------------------------------------------------- |
| Nvidia GPU | ✅                 | ✅                 | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)       |
| Ascend NPU | ✅                 | ✅                 | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)      |
| Apple SOC  | ✅                 | ✅                 | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)        |
| Cambricon MLU | ✅                 | ✅                 | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅                 | ✅                 | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅                 | ✅                 | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU | ✅                 | ✅                 | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅                 | ✅                 | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU          | ✅                 | ✅                 | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)                |
| Memory       | ✅                 | ✅                 | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)             |
| Disk         | ✅                 | ✅                 | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)               |
| Network      | ✅                 | ✅                 | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)         |

Feel free to submit an issue or PR if you want to add support for other hardware!

<br>

## Framework Integrations

Combine your favorite frameworks with SwanLab!  Here's a list of integrated frameworks; submit an [Issue](https://github.com/swanhubx/swanlab/issues) to request integrations.

**Basic Frameworks**
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

## Plugins & API

Enhance your experiment management experience by extending SwanLab's capabilities with plugins!

-   [Customize Your Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email Notification](https://docs.swanlab.cn/plugin/notification-email.html)
-   [Feishu Notification](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notification](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [WeChat Work Notification](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord Notification](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack Notification](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [File Logger](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open APIs:
- [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparison with Similar Tools

### Tensorboard vs SwanLab

*   **Cloud Support:** SwanLab offers cloud-based synchronization and storage for your training experiments, allowing you to view training progress remotely, manage historical projects, share experiment links, send real-time notifications, and view experiments on multiple devices. Tensorboard is an offline tool.
*   **Collaboration:** SwanLab allows easy management of training projects by multiple people, sharing experiment links, and cross-space communication. Tensorboard is designed primarily for individuals.
*   **Persistent, Centralized Dashboard:** Regardless of where you train your model, your results are recorded in the same centralized dashboard. TensorBoard requires effort to copy and manage TFEvent files from different machines.
*   **Powerful Tables:** SwanLab tables help you view, search, and filter results from different experiments. TensorBoard is not suitable for large projects.

### Weights and Biases vs SwanLab

*   Weights and Biases is a closed-source MLOps platform that requires internet access.
*   SwanLab supports online use, along with open-source, free, and self-hosted versions.

<br>

## Community

### Related Repositories

-   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosted deployment scripts repository
-   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation repository
-   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline Dashboard repository

### Community and Support

-   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report errors and issues with SwanLab
-   [Email Support](zeyi.lin@swanhub.co): Report any issues with SwanLab
-   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discuss issues with SwanLab, share AI tech

### SwanLab README Badges

Add SwanLab badges to your README to showcase your use of SwanLab:

[![][tracking-swanlab-shield]][tracking-swanlab-shield-link]、[![][visualize-swanlab-shield]][visualize-swanlab-shield-link]

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Your Publications

If SwanLab has been helpful in your research, cite us using the following format:

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

Interested in contributing?  Review our [Contributing Guide](CONTRIBUTING.md).

We also welcome your support by sharing SwanLab on social media, at events, and in conferences!

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

<!-- link -->

[release-shield]: https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square
[release-link]: https://github.com/swanhubx/swanlab/releases

[license-shield]: https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square
[license-shield-link]: https://