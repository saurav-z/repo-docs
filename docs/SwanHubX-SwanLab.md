<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

## SwanLab: Supercharge Your AI Training with Open-Source Experiment Tracking and Visualization

**SwanLab is your all-in-one solution for streamlining machine learning workflows, offering robust experiment tracking, insightful visualizations, and collaborative features for researchers and teams.**  Effortlessly integrate with 30+ frameworks, visualize training progress, and share your results. 

<a href="https://swanlab.cn">🔥SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Documentation</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks</a>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
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

<br>

![SwanLab Overview](readme_files/swanlab-overview.png)

[中文](README.md) / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>

<br/>

## Key Features

*   **Experiment Tracking & Visualization:**
    *   Track key metrics, hyperparameters, and metadata with minimal code.
    *   Visualize training progress with interactive charts and graphs.
    *   Support for scalar metrics, images, audio, text, videos, 3D point clouds, and custom ECharts.
    *   Visualizations for LLM content, including Markdown rendering.
*   **Comprehensive Framework Integrations:**
    *   Seamless integration with 30+ popular frameworks, including PyTorch, 🤗Hugging Face Transformers, PyTorch Lightning, LLaMA Factory, and more.
*   **Hardware Monitoring:**
    *   Real-time monitoring of CPU, GPU (Nvidia, Ascend, Intel, Moore Threads, etc.), and memory usage.
*   **Experiment Management:**
    *   Centralized dashboard for managing projects and experiments.
    *   Intuitive interface for comparing results and identifying trends.
*   **Collaboration & Sharing:**
    *   Share experiments with persistent URLs.
    *   Facilitate collaborative training within teams.
*   **Self-Hosted & Offline Support:**
    *   Use SwanLab offline or on your own infrastructure with the self-hosted community version.

<br/>

## Getting Started

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .
```

</details>

<details><summary>Install Offline Dashboard</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login & Get API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Get your API Key from [API Key Settings](https://swanlab.cn/settings).
3.  Open your terminal and run:

```bash
swanlab login
```

Enter your API key when prompted.

### 3. Integrate with Your Code

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

View your experiment on [SwanLab](https://swanlab.cn).

<br>

## Self-Hosting

Deploy a self-hosted community edition to view your dashboard offline.

![SwanLab Docker](./readme_files/swanlab-docker.png)

### 1. Deploy with Docker

See [Docker Deployment Guide](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html) for details.

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Quick Installation (China):

```bash
./install.sh
```

DockerHub Installation:

```bash
./install-dockerhub.sh
```

### 2. Point Experiments to Self-Hosted Service

Login to your self-hosted instance:

```bash
swanlab login --host http://localhost:8000
```

Then, your experiments will be logged to your self-hosted instance.

<br>

## Examples & Tutorials

*   **[MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)**
*   **[FashionMNIST Classification](https://docs.swanlab.cn/examples/fashionmnist.html)**
*   **[Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)**
*   **[Resnet Cats vs. Dogs Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)**
*   **[YOLO Object Detection](https://docs.swanlab.cn/examples/yolo.html)**
*   **[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)**

🌟  Submit a PR if you want to add tutorials!

<br>

## Hardware Recording

SwanLab records **hardware information** and **resource usage** during AI training.

See the hardware support table:

| Hardware       | Information Recording | Resource Monitoring | Script                                                               |
| -------------- | --------------------- | ------------------- | -------------------------------------------------------------------- |
| NVIDIA GPU     | ✅                    | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU     | ✅                    | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC      | ✅                    | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU  | ✅                    | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU  | ✅                    | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅                    | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU      | ✅                    | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU      | ✅                    | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU            | ✅                    | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory         | ✅                    | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk           | ✅                    | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network        | ✅                    | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Submit an Issue or PR to record more hardware!

<br>

## Framework Integrations

Use SwanLab with your favorite frameworks!  
Here is a list of frameworks integrated:

**Core Frameworks**
- [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
- [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
- [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specific/Fine-tuning Frameworks**
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

**Other Frameworks**
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

Enhance your experiment management with SwanLab plugins!

- [Customize your plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
- [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
- [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
- [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
- [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
- [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
- [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
- [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
- [FileLogDir Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:
- [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparison with Similar Tools

### TensorBoard vs. SwanLab

*   **Cloud-Based Support:** SwanLab offers seamless cloud synchronization and experiment storage for convenient remote access, project management, experiment sharing, real-time notifications, and multi-device viewing. TensorBoard is a local experiment tracking tool.
*   **Multi-User Collaboration:** Facilitate training projects and share experiments, collaborate across teams. TensorBoard is designed for individual use.
*   **Persistent, Centralized Dashboard:** All results are logged to the same dashboard regardless of where your model trains.
*   **Enhanced Table View:** Search, filter results from different experiments, and quickly identify the best-performing models.  TensorBoard is not suitable for large projects.

### Weights & Biases vs. SwanLab

*   Weights & Biases is a closed-source MLOps platform.
*   SwanLab supports cloud and open-source, free, self-hosted versions.

<br>

## Community

### Related Repositories

-   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official Documentation
-   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Web code for the offline dashboard.
-   [self-hosted](https://github.com/swanhubx/self-hosted): Scripts for self-hosting.

### Community & Support

-   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): For errors and issues.
-   [Email Support](zeyi.lin@swanhub.co): For questions.
-   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discuss and share AI technology.

### SwanLab README Badges

Add SwanLab badges to your README:

[![Track with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Publications

If SwanLab helps your research, cite it using:

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

Read the [contribution guidelines](CONTRIBUTING.md).

Support SwanLab by sharing on social media, events, and conferences - Thank you!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## License

Licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)