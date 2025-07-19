<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br/>

# SwanLab: Open-Source Deep Learning Experiment Tracking & Visualization

SwanLab is a powerful, open-source tool designed to track, visualize, and collaborate on your deep learning experiments, offering a modern and user-friendly interface.  Visit the original repository: [SwanLab](https://github.com/SwanHubX/SwanLab)

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

[English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking & Visualization**: Visualize your training metrics, hyperparameters, and model artifacts with an intuitive UI, including:
    *   📈 **Rich Charting**: Create and customize line charts, media plots (images, audio, text, video), 3D point clouds, biochemical molecules, and custom charts with Echarts.
    *   📝 **LLM Content Visualization**:  Dedicated charts for LLM training, supporting Markdown rendering.
    *   ⚙️ **Automatic Logging**: Logs metrics, hardware environment, Git repository, Python environment, and project details.
    *   🔄 **Resume Training**: Supports resuming training and appending new data to existing experiments.

*   **Framework Integrations**: Seamlessly integrate with popular deep learning frameworks, including:
    *   PyTorch
    *   Hugging Face Transformers
    *   PyTorch Lightning
    *   And many more (over 30 supported).

*   **Hardware Monitoring**: Real-time monitoring of hardware metrics:
    *   CPU, Ascend NPU, Nvidia GPU,  Cambricon MLU,  Kunlunxin XPU, Hygon DCU,  Metax GPU, Moore Threads GPU, Memory.

*   **Experiment Management**: Centralized dashboard for managing projects and experiments.

*   **Comparison & Collaboration**: Compare hyper-parameters and results using online tables and comparison charts, and collaborate with your team.

*   **Online Collaboration**: Share and collaborate on experiments within your team, making it easier to review and comment on results.

*   **Experiment Sharing**: Easily share experiments with persistent URLs.

*   **Self-Hosting**: Supports offline usage and self-hosted deployments.

*   **Plugin Extensibility**: Extend functionality with plugins for notifications, CSV writing, file logging and more.

>  **Star the project on GitHub to stay updated!** ⭐️

![star-us](readme_files/star-us.png)

<br/>

## Recent Updates

*   **2025.07.17**: 📊 Enhanced line chart configuration with flexible customization options. Support for GIF files via `swanlab.Video`.
*   **2025.07.10**: 📚 Improved text views with Markdown rendering,  and support for arrow key navigation, created via `swanlab.echarts.table` and `swanlab.Text`.
*   **2025.07.06**: 🚄 Resume training support, new file logger plugin, and integration with the Ray and ROLL frameworks.
*   **2025.06.27**: 📊 Small line chart zoom, single line chart smoothing, and improved image chart interaction.
*   **2025.06.20**: 🤗 Integrated the Accelerate framework.
*   **2025.06.18**: 🐜 Integrated the AREAL framework.
*   **2025.06.11**: 📊 Support for `swanlab.echarts.table`,  group stretching for better chart display, and table enhancements.
*   **2025.06.08**: ♻️ Local storage of experiment logs, uploading local logs to the cloud/self-hosted instances, and  Hygon DCU support.
*   **2025.06.01**: 🏸 Chart dragging and ECharts custom charts.
<details><summary>Show More Updates</summary>

*   **2025.05.25**:  Standard error stream logging support, hardware monitoring for Moore Threads, added security features for running commands.
*   **2025.05.14**: Support for experiment tags, line chart log scale, group dragging, improved metrics upload, and `swanlab.OpenApi`.
*   **2025.05.09**: Line chart creation, data source selection for charts, and GitHub badge generation.
*   **2025.04.23**: Line chart editing, Kunlunxin XPU hardware monitoring.
*   **2025.04.11**: Line chart region selection, global step range for line charts, and hide all charts functionality.
*   **2025.04.08**:  `swanlab.Molecule` data type support and table view state persistence.
*   **2025.04.07**:  Integrated with EvalScope.
*   **2025.03.30**:  `swanlab.Settings` for refined experiment control, Cambricon MLU hardware monitoring, and Slack/Discord notifications.
*   **2025.03.21**: 🤗 HuggingFace Transformers integration (>=4.50.0), new  Object3D chart support, and improved GPU memory, disk, and network monitoring.
*   **2025.03.12**:  🎉🎉 SwanLab private deployment version released and support for  plugin extensions.
*   **2025.03.09**: Experiment sidebar widening and git code button and `sync_mlflow` function.
*   **2025.03.06**:  Integrated with DiffSynth Studio.
*   **2025.03.04**:  MLFlow conversion,
*   **2025.03.01**: Added experiment movement.
*   **2025.02.24**:  Integrated with EasyR1.
*   **2025.02.18**: Integrated with Swift.
*   **2025.02.16**: Added Grouping/Creating group functionality.
*   **2025.02.09**: Integrated with veRL.
*   **2025.02.05**:  `swanlab.log` supports nested dictionaries, name and notes parameters.
*   **2025.01.22**: Added  `sync_tensorboardX` and `sync_tensorboard_torch`.
*   **2025.01.17**: Added `sync_wandb` and improved logging performance.
*   **2025.01.11**: Performance improvements for cloud project tables.
*   **2025.01.01**: Line chart smoothing, dragging, and size change improvements.
*   **2024.12.22**: Integrated with LLaMA Factory.
*   **2024.12.15**:  Hardware monitoring (0.4.0) for CPU, Ascend NPU, and Nvidia GPU.
*   **2024.12.06**: Integration with LightGBM and XGBoost, and increased log line length.
*   **2024.11.26**: Hardware tab hardware detection and QingCloud support.
</details>

<br/>

## Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

```bash
# Option 1:
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Option 2:
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Extension</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login & Get API Key

1.  [Sign up](https://swanlab.cn) for a free account.

2.  Log in and copy your API Key from User Settings -> [API Key](https://swanlab.cn/settings).

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

You're all set!  View your experiment on [SwanLab](https://swanlab.cn).

<br>

## Self-Hosting

Self-hosted community versions provide an offline experience for viewing the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploying a self-hosted version using Docker

Refer to the documentation: [Docker Deployment](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

China's fast installation:

```bash
./install.sh
```

Install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Assign experiments to self-hosted services

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Once logged in, you can record your experiments on the self-hosted service.

<br>

## Usage Examples & Tutorials

Comprehensive examples are available to help you get started.

**Open Source Tutorial Projects Using SwanLab:**
*   [happy-llm](https://github.com/datawhalechina/happy-llm) ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm) ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek) ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)

**Papers Using SwanLab:**
*   [Animation Needs Attention](https://arxiv.org/abs/2507.03916)

**Tutorials:**
*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Image Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Finetuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Finetuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Finetuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multi-modal Large Model Finetuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multi-modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)

<br>

## Hardware Monitoring

SwanLab records **hardware information** and **resource usage** during AI training.

| Hardware        | Information Recorded | Resource Monitoring | Script                                                                                 |
| --------------- | -------------------- | ------------------- | -------------------------------------------------------------------------------------- |
| Nvidia GPU      | ✅                  | ✅                 | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU      | ✅                  | ✅                 | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC       | ✅                  | ✅                 | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU   | ✅                  | ✅                 | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU   | ✅                  | ✅                 | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅                  | ✅                 | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU       | ✅                  | ✅                 | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU       | ✅                  | ✅                 | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU             | ✅                  | ✅                 | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)       |
| Memory          | ✅                  | ✅                 | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)  |
| Disk            | ✅                  | ✅                 | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)  |
| Network         | ✅                  | ✅                 | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Feel free to submit issues and PRs to support additional hardware!

<br>

## Framework Integrations

Integrate your favorite framework with SwanLab!

Here's a list of currently integrated frameworks.  Submit an [Issue](https://github.com/swanhubx/swanlab/issues) to request integration.

**Base Frameworks**

*   PyTorch
*   MindSpore
*   Keras

**Specialized/Finetuning Frameworks**

*   PyTorch Lightning
*   HuggingFace Transformers
*   LLaMA Factory
*   Modelscope Swift
*   DiffSynth Studio
*   Sentence Transformers
*   PaddleNLP
*   OpenMind
*   Torchtune
*   XTuner
*   MMEngine
*   FastAI
*   LightGBM
*   XGBoost

**Evaluation Frameworks**

*   EvalScope

**Computer Vision**

*   Ultralytics
*   MMDetection
*   MMSegmentation
*   PaddleDetection
*   PaddleYOLO

**Reinforcement Learning**

*   Stable Baseline3
*   veRL
*   HuggingFace trl
*   EasyR1
*   AReaL
*   ROLL

**Other Frameworks:**

*   Tensorboard
*   Weights&Biases
*   MLFlow
*   HuggingFace Accelerate
*   Ray
*   Unsloth
*   Hydra
*   Omegaconf
*   OpenAI
*   ZhipuAI

[More integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br>

## Plugins & API

Extend SwanLab's capabilities with plugins to enhance your experiment management!

*   [Custom Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
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

## Comparisons

### TensorBoard vs. SwanLab

*   **☁️ Online Support**: SwanLab syncs experiments online for remote access, project management, and easy sharing. TensorBoard is primarily a local tool.
*   **👥 Collaboration**: SwanLab is built for collaboration within teams, with features for shared projects, experiment sharing, and team discussions. TensorBoard is primarily for individual use.
*   **💻 Persistent Dashboard**:  Your results are recorded in a single centralized dashboard regardless of where you're training.
*   **💪 Powerful Tables**: SwanLab's tables allow users to easily compare, search, filter, and view results from different experiments.

### Weights & Biases vs SwanLab

*   Weights & Biases is a closed-source, cloud-based MLOps platform.
*   SwanLab offers open-source, free, and self-hosted options, giving you more flexibility and control.

<br>

## Community

### Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Documentation repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard repository
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting deployment scripts repository

### Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report bugs and issues.
*   [Email](zeyi.lin@swanhub.co): For feedback on SwanLab usage.
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Discuss usage and AI technology.

### SwanLab README Badges

Add a SwanLab badge to your README:

[![Track with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab

If you find SwanLab helpful, please cite it in your publications using the following format:

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

Read the [Contribution Guide](CONTRIBUTING.md) before contributing.

We welcome your support through social media, events, and conferences!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## License

SwanLab is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)