<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## SwanLab: Your All-in-One Solution for Deep Learning Experiment Tracking and Visualization

**SwanLab is an open-source, modern tool designed to revolutionize your deep learning workflow, offering comprehensive experiment tracking, visualization, and collaboration features.**  Seamlessly integrate SwanLab with your code and monitor your experiments whether you're working locally or in the cloud.  [Explore the original repository](https://github.com/SwanHubX/SwanLab) for more details.

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)

<br/>

![](readme_files/swanlab-overview.png)

**Languages:** [中文](README_CN.md) / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking and Visualization:**
    *   Track metrics, hyperparameters, and system resources (CPU, GPU, Memory, Disk, Network) with easy-to-use Python APIs.
    *   Visualize training progress with interactive charts (line charts, media, 3D point clouds, Echarts custom charts, LLM content visualization).
    *   Comprehensive metadata support: scalar metrics, images, audio, text, video, 3D point clouds, biochemical molecules, and custom charts.
    *   Supports Log scale on line charts
    *   Supports resume of training.
*   **Flexible Deployment Options:**
    *   **Cloud Support:** Track your experiments on the cloud, and access your training progress remotely from any device.
    *   **Self-Hosted:** Run SwanLab locally or on your own servers with full functionality, ideal for offline environments.
    *   **Integration with popular frameworks**: Integrate seamlessly with over 30 popular frameworks, including PyTorch, Hugging Face Transformers, PyTorch Lightning, and many more.
*   **Hardware Monitoring:**
    *   Monitor your system's hardware resources, including:
        *   Nvidia GPUs
        *   Ascend NPUs
        *   Apple SOCs
        *   Cambricon MLUs
        *   Kunlunxin XPUs
        *   Moore Threads GPUs
        *   MetaX GPUs
        *   Hygon DCUs
        *   CPUs
        *   Memory
        *   Disk
        *   Network
*   **Experiment Management & Collaboration:**
    *   Centralized dashboard for managing projects and experiments.
    *   Compare experiments to identify patterns and optimize your models.
    *   Enable collaboration within your team and share results easily.
*   **Other Features:**
    *   Experiment Tagging.
    *   Support for a growing list of integrations, including [MLFlow](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html), [Weights & Biases](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html) and many more.
    *   Plugin support for extending SwanLab's functionality, such as [CSV writer](https://docs.swanlab.cn/plugin/writer-csv.html) and [Slack notifications](https://docs.swanlab.cn/plugin/notification-slack.html).

<br/>

## Recent Updates

*   **[2025.08.19]** Improved chart rendering and low-intrusive loading animations. Integrated with the [MLX-LM](https://github.com/ml-explore/mlx-lm) and [SpecForge](https://github.com/sgl-project/SpecForge) frameworks.
*   **[2025.08.06]** Introduced training collaboration features, workspace list view with project tags
*   **[2025.07.29]** Enhanced experiment filtering and sorting, table view column customization, multi-API key management, and new charts (PR curve, ROC curve, confusion matrix)
*   **[2025.07.17]** Improved line chart configuration, support for swanlab.Video, and global chart dashboard improvements.
*   **[2025.07.10]** Enhanced text view with Markdown rendering and support for swanlab.echarts.table and swanlab.Text.
*   **[2025.07.06]** Added resume from a breakpoint, File Recorder plugin, and integration with the [Ray](https://github.com/ray-project/ray) and [ROLL](https://github.com/volcengine/ROLL) frameworks.

<details><summary>Complete Update Log</summary>

... (See original README for full update log)

</details>

<br>

## Online Demos

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |

| [EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br/>

## Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .
```

</details>

<details><summary>Offline Dashboard Extension</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Login to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open a terminal and run:

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

<br/>

## Self-Hosting

Support offline viewing of SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted Version with Docker

Instructions: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For fast installation in China:

```bash
./install.sh
```

To install by pulling images from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point Experiments to Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After login, all experiments will be recorded to your self-hosted service.

<br/>

## Real-World Use Cases & Tutorials

See [examples](https://docs.swanlab.cn/examples/) for MNIST, FashionMNIST, Cifar10, audio, object detection, GAN training, and more!

### Tutorials:
* [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   and many more (listed in the original README)

🌟 Submit a PR to add your tutorial!

<br>

## 🎮 Hardware Monitoring

Hardware Monitoring is available for:

| Hardware        | Information Recorded | Resource Monitoring | Script                                                               |
| :-------------- | :------------------- | :------------------ | :------------------------------------------------------------------- |
| Nvidia GPU      | ✅                  | ✅                 | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)          |
| Ascend NPU      | ✅                  | ✅                 | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)          |
| Apple SOC       | ✅                  | ✅                 | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)          |
| Cambricon MLU   | ✅                  | ✅                 | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py)          |
| Kunlunxin XPU   | ✅                  | ✅                 | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py)        |
| Moore Threads GPU | ✅                  | ✅                 | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU       | ✅                  | ✅                 | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py)           |
| Hygon DCU       | ✅                  | ✅                 | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py)           |
| CPU             | ✅                  | ✅                 | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)                   |
| Memory          | ✅                  | ✅                 | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)               |
| Disk            | ✅                  | ✅                 | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)                   |
| Network         | ✅                  | ✅                 | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)            |

<br>

## 🚗 Framework Integrations

Integrate your favorite framework with SwanLab!

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

## 🔌 Plugins & API

Extend SwanLab's functionality using plugins for enhanced experiment management!

- [Custom Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
- [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
- [Lark Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
- [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
- [WeCom Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
- [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
- [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
- [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
- [FileLogDir Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:
- [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparison with Familiar Tools

*   **Tensorboard vs SwanLab**
    *   SwanLab offers cloud support for remote access and collaboration.
    *   SwanLab is designed for team-based machine learning efforts.
    *   SwanLab provides a persistent and centralized dashboard.
    *   SwanLab's tables are more robust for complex data analysis.

*   **Weights and Biases vs SwanLab**
    *   SwanLab is open-source, free, and allows self-hosting, unlike the Weights and Biases platform.

<br>

## 👥 Community

### Related Repositories

-   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting scripts repository
-   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation repository
-   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard repository

### Community & Support

-   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report bugs and ask questions.
-   [Email Support](zeyi.lin@swanhub.co): For feedback and questions.
-   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discuss SwanLab and the latest AI technology.

### SwanLab README Badges

Add SwanLab badges to your README:

[![][tracking-swanlab-shield]][tracking-swanlab-shield-link]  
[![][visualize-swanlab-shield]][visualize-swanlab-shield-link]

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

Find more design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Publications

If SwanLab has helped your research, please cite it using this format:

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

Interested in contributing? Read the [contribution guide](CONTRIBUTING.md).

Also, support SwanLab by sharing on social media, at events, and during conferences!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

<br/>

## 📃 License

This repository is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)