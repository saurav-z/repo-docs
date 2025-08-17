<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

**Supercharge Your AI Experiment Tracking: SwanLab - the Open-Source, Modern Visualization Tool for Deep Learning.**

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
[![Wechat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

![](readme_files/swanlab-overview.png)

[中文 / English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>

<br/>

## Table of Contents

*   [🌟 Recent Updates](#-recent-updates)
*   [👋🏻 What is SwanLab?](#-what-is-swanlab)
*   [📃 Online Demo](#-online-demo)
*   [🏁 Quickstart](#-quickstart)
*   [💻 Self-Hosting](#-self-hosting)
*   [🔥 Use Cases](#-use-cases)
*   [🎮 Hardware Monitoring](#-hardware-monitoring)
*   [🚗 Framework Integrations](#-framework-integrations)
*   [🔌 Plugins and API](#-plugins-and-api)
*   [🆚 Comparisons with Similar Tools](#-comparisons-with-similar-tools)
*   [👥 Community](#-community)
*   [📃 License](#-license)
*   [Star History](#star-history)

<br/>

## 🌟 Recent Updates

*   **[List of Recent Updates]**  *See the original README for the detailed changelog.*

<br/>

## 👋🏻 What is SwanLab?

SwanLab is an open-source, user-friendly tool designed for tracking, visualizing, and collaborating on your deep learning experiments. It simplifies the process of monitoring your model training, offering a comprehensive platform for researchers and engineers.  SwanLab empowers you to understand your experiments more effectively, compare results, and collaborate seamlessly.

**Key Features:**

*   **📊 Experiment Metrics & Hyperparameter Tracking:** Easily track key metrics and hyperparameters within your ML pipeline using a simple Python API.
    *   ☁️ **Cloud Support:** Use SwanLab in the cloud for easy access and collaboration.
    *   📝 **Hyperparameter and Metric Summarization:** Summarize and analyze key parameters, metrics and tabular data.
    *   🌸 **Training Visualization:** Gain insights into your experiments with interactive visualizations.

*   **📈 Rich Data Types:**  Supports a variety of data types, including:
    *   Scalar metrics, images, audio, text, videos, 3D point clouds, biochemical molecules, custom ECharts charts, and more.

*   **🖼️ Comprehensive Charting:** Visualize data with a wide range of chart types, including:
    *   Line charts, media charts (images, audio, text, video), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, and custom charts.

*   **🚀 Automated Logging:** Automatically logs essential data, including:
    *   Logs, hardware environment, Git repository information, Python environment details, Python library listings, and project runtime directories.

*   **💾 Resume Training Support:**  Resume training and continue logging metrics even after interruption, enabling more flexibility in training runs.

*   **⚡️ Extensive Framework Integrations:** Seamlessly integrates with over 30 popular frameworks, including: PyTorch, Hugging Face Transformers, PyTorch Lightning, LLaMA Factory,  MMDetection, Ultralytics, PaddleDetetion, LightGBM, XGBoost, Keras, Tensorboard, Weights&Biases, OpenAI and others.

*   **💻 Hardware Monitoring:** Monitor system-level hardware metrics in real-time for CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (Muxi XPU), Moore Threads GPU, memory, and others.

*   **📦 Experiment Management:** Centralized dashboard designed for training, enabling you to manage multiple projects and experiments with a global overview.

*   **🆚 Result Comparison:** Compare hyperparameters and results across experiments using online tables and comparison charts for better insights.

*   **👥 Online Collaboration:** Collaborate with your team in real-time, sync experiments within a project, and exchange feedback with online views.

*   **✉️ Share Results:** Share experiment results easily with shareable URLs that can be included in notes and shared with partners.

*   **💻 Self-Hosting Support:** Use the offline dashboard to view and manage your experiments, even in offline environments.

*   **🔌 Plugin Extensibility:** Extend SwanLab's functionality with plugins, e.g. [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html), [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html), [CSV Writers](https://docs.swanlab.cn/plugin/writer-csv.html), etc.

> \[!IMPORTANT]
>
> **Star the project** to receive notifications about new releases on GitHub! ⭐️

![star-us](readme_files/star-us.png)

<br/>

## 📃 Online Demo

Explore SwanLab's capabilities with the following online demos:

*   **[Demo 1: ResNet50 Cat/Dog Classification][demo-cats-dogs]** - Track a simple ResNet50 model's training on a cat/dog image classification task.
*   **[Demo 2: YOLOv8 Object Detection on COCO128][demo-yolo]** - Track hyperparameter and metrics for a YOLOv8 object detection model on the COCO128 dataset.
*   **[Demo 3: Qwen2 Instruction Fine-tuning][demo-qwen2-sft]** - Track the instruction fine-tuning of the Qwen2 large language model.
*   **[Demo 4: LSTM Google Stock Prediction][demo-google-stock]** - Use an LSTM model to predict future Google stock prices.
*   **[Demo 5: ResNeXt101 Audio Classification][demo-audio-classification]** - A deep dive into audio classification using ResNeXt101.
*   **[Demo 6: Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl]** - Fine-tuning Qwen2-VL multi-modal large language model.
*   **[Demo 7: EasyR1 Multi-Modal LLM RL Training][demo-easyr1-rl]** - RL training with the EasyR1 framework.
*   **[Demo 8: Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo]** - GRPO training based on the Qwen2.5-0.5B model in the GSM8k dataset.

[More examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br/>

## 🏁 Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

Install SwanLab to experience latest features:

```bash
# Method 1:
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2:
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Install Offline Dashboard Extension</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login & API Key

1.  [Register](https://swanlab.cn) for free.

2.  Log in and copy your API key from User Settings > [API Key](https://swanlab.cn/settings).

3.  In your terminal:

```bash
swanlab login
```

Enter your API key when prompted.

### 3. Integrate SwanLab with Your Code

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

Visit [SwanLab](https://swanlab.cn) to view your experiment.

<br/>

## 💻 Self-Hosting

Self-hosting Community Version supports viewing SwanLab Dashboard Offline.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy a Self-Hosted Version with Docker

Details can be found at: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For a quick installation in China:

```bash
./install.sh
```

Or, pull images from DockerHub to install:

```bash
./install-dockerhub.sh
```

### 2. Point Experiments to Your Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After login, experiment results will be logged in the self-hosted service.

<br/>

## 🔥 Use Cases

**Open-Source Projects using SwanLab:**

*   *List of Open Source Project Names and Links (Example: [happy-llm](https://github.com/datawhalechina/happy-llm))*
    *   ...

**Research Papers using SwanLab:**

*   *List of Paper Titles and Links (Example: [Animation Needs Attention...](https://arxiv.org/abs/2507.03916))*
    *   ...

**Tutorials:**

*   *Links to Various Tutorial Pages (Example: [MNIST Hand-written Digit Recognition](https://docs.swanlab.cn/examples/mnist.html))*
    *   ...

🌟 Contributions of tutorials are welcomed!

<br/>

## 🎮 Hardware Monitoring

SwanLab monitors hardware information and resource usage during AI training. Supported hardware:

| Hardware           | Information Recording | Resource Monitoring | Script                                                                     |
| :----------------- | :-------------------- | :------------------- | :------------------------------------------------------------------------- |
| NVIDIA GPU         | ✅                    | ✅                   | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU         | ✅                    | ✅                   | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC          | ✅                    | ✅                   | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU      | ✅                    | ✅                   | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU      | ✅                    | ✅                   | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU  | ✅                    | ✅                   | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Muxi GPU           | ✅                    | ✅                   | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU          | ✅                    | ✅                   | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU                | ✅                    | ✅                   | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)          |
| Memory             | ✅                    | ✅                   | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)   |
| Disk               | ✅                    | ✅                   | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)       |
| Network          | ✅                    | ✅                   | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)       |

Open an Issue or PR to record other hardware!

<br/>

## 🚗 Framework Integrations

Use your favorite frameworks with SwanLab!

Below is a list of integrated frameworks. Open a [Issue](https://github.com/swanhubx/swanlab/issues) to request integrations.

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

<br/>

## 🔌 Plugins and API

Enhance your SwanLab experience with plugins!

*   [Customize your plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [Enterprise WeChat Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br/>

## 🆚 Comparisons with Similar Tools

### TensorBoard vs SwanLab

*   **☁️ Online Usage:** SwanLab enables easy cloud-based experiment synchronization and saving, allowing you to remotely track training progress, manage projects, share experiment links, send real-time notifications, and view experiments on multiple devices. TensorBoard is a purely offline experiment tracking tool.
*   **👥 Collaboration:** SwanLab streamlines collaboration for multi-person, cross-team ML projects. With SwanLab you can readily manage training projects, share experiment links, and share insights. TensorBoard is primarily designed for individual use and has limited support for collaboration.
*   **💻 Persistent, Centralized Dashboard:** Your results are logged in a single dashboard, regardless of where you train your models. TensorBoard requires time-consuming TFEvent file management.
*   **💪 Advanced Tables:** Use SwanLab tables to view, search, and filter results from different experiments.
  TensorBoard isn't suitable for large projects.

### Weights and Biases vs SwanLab

*   Weights and Biases is a closed-source MLOps platform that requires internet connectivity.
*   SwanLab offers both cloud and self-hosted versions that are open source and free.

<br/>

## 👥 Community

### Ecosystem

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard web code
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting scripts

### Community & Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Issues, bugs, and feature requests
*   [Email Support](zeyi.lin@swanhub.co): Feedback or questions
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): For discussion and support

### SwanLab README Badges

Add these badges to your project's README:

[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

Find more design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Publications

If you find SwanLab useful, please cite it using the following format:

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

Read the [Contribution Guidelines](CONTRIBUTING.md) before contributing.

Support SwanLab by sharing through social media, activities, and conferences, and we'll be very grateful!

<br/>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br/>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## 📃 License

This repository is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)