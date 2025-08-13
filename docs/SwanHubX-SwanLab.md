<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

**Supercharge your machine learning workflow with SwanLab, the open-source, modern tool for experiment tracking and visualization.**

<a href="https://swanlab.cn">🔥SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Documentation</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report an Issue</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks</a>

[![GitHub Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
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

![](readme_files/swanlab-overview.png)

[English](README_EN.md) / [中文](README.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>

<br/>

## Key Features

*   **Experiment Tracking and Visualization**:  Visualize your training metrics, hyperparameters, and other data with interactive charts and tables.  Monitor training progress in real-time.

*   **Framework Integrations**:  Seamlessly integrate with 30+ popular machine learning frameworks, including PyTorch, TensorFlow, and others.

*   **Hardware Monitoring**:  Track CPU, GPU (Nvidia, Ascend, etc.), memory, and disk utilization to optimize resource usage.

*   **Experiment Management**:  Organize and compare your experiments with an intuitive dashboard.

*   **Collaboration**:  Share your experiments with your team and collaborate effectively.

*   **Cloud and Self-Hosted Options**:  Use SwanLab online or self-host it for offline access and data privacy.

*   **Customization**:  Extend SwanLab's functionality with plugins for notifications, custom charts, and more.

## What is SwanLab?

SwanLab is an open-source, modern tool designed to streamline your deep learning workflow. It provides a comprehensive platform for tracking, visualizing, and collaborating on your machine learning experiments, making it easier to reproduce and improve your models.

*   **Experiment Tracking and Visualization:** SwanLab provides an intuitive UI to track and visualize key metrics, helping you understand the training process.

*   **Framework Integration:** SwanLab is integrated with various popular ML frameworks, which can be used in a variety of tasks.

*   **Hardware Monitoring:** Real-time hardware metrics help you identify potential bottlenecks and optimize resource allocation.

*   **Experiment Comparison:** SwanLab allows you to easily compare different experiments to identify the best-performing models.

*   **Collaboration**: Facilitates team collaboration through shared experiment dashboards and result sharing.

*   **Self-Hosting**: SwanLab can be self-hosted for offline use and data privacy.

*   **Extend with Plugins**: Plugin functionality allows you to extend SwanLab to your needs.

Here's what SwanLab can help with:

*   **Experiment Tracking**: Easily track metrics, hyperparameters, and more during your training runs.
*   **Visualization**: Visualize your experiments with interactive charts and dashboards.
*   **Collaboration**: Share your experiments with your team and collaborate effectively.
*   **Reproducibility**: Easily reproduce your experiments with the help of the organized data.
*   **Insight and Optimization**: Analyze training progress and quickly understand what is affecting your results.
*   **Cloud or Local**: SwanLab provides cloud-based tracking and local-host tracking, to make the experiment available in all environments.

## Recent Updates

*   **Training Collaboration** (Aug 6, 2025): Project collaborators support, list view in the workspace.
*   **Experiment Filtering and Sorting** (Jul 29, 2025): Sidebar filtering and sorting, table view column control panel.
*   **Multi-API Key Management** (Jul 29, 2025): Enhanced data security.
*   **PR/ROC Curves** (Jul 29, 2025): New chart support.
*   **More Flexible Configuration** (Jul 17, 2025): Configurable chart line type, color, and legend positioning.
*   **Video Support** (Jul 17, 2025): GIF file logging and visualization.
*   **Resume Training** (Jul 06, 2025): Support for resuming training from checkpoints.
*   **File Recorder** (Jul 06, 2025): New plugin.
*   **Echarts Table** (Jun 11, 2025): Data type for text chart display.

<details><summary>Full Changelog</summary>

*   **(June 08, 2025):** Local Log Persistence, Ascend Hardware Monitoring.
*   **(June 01, 2025):** Chart Drag and Drop, ECharts Custom Charts, MX GPU Support.
*   **(May 25, 2025):** Standard Error Stream Logging, Moore Threads GPU Support.
*   **(May 14, 2025):** Experiment Tags, Log Scale, Group Dragging.
*   **(May 09, 2025):** Line Chart Creation, Training Project GitHub Badges.
*   **(April 23, 2025):** Line Chart Editing, Kunlunxin XPU Hardware Detection.
*   **(April 11, 2025):** Line Chart Region Selection, Step Range Selection.
*   **(April 08, 2025):** swanlab.Molecule Data Type, Table View State Preservation.
*   **(April 07, 2025):** EvalScope Integration.
*   **(March 30, 2025):** swanlab.Settings Method, Cambricon MLU Hardware Monitoring, Slack and Discord Notifications.
*   **(March 21, 2025):** Hugging Face Transformers Integration, Object3D Chart, GPU Memory, Disk Usage, and Network Monitoring.
*   **(March 12, 2025):** Private Deployment Release, Plugin Support.
*   **(March 09, 2025):** Experiment Sidebar Expansion, Git Code Button, sync\_mlflow.
*   **(March 06, 2025):** DiffSynth Studio Integration.
*   **(March 04, 2025):** MLFlow Conversion.
*   **(March 01, 2025):** Move Experiment Feature.
*   **(February 24, 2025):** EasyR1 Integration.
*   **(February 18, 2025):** Swift Integration.
*   **(February 16, 2025):** Chart Grouping.
*   **(February 09, 2025):** veRL Integration.
*   **(February 05, 2025):** Nested Dictionary Logging, Name and Notes Parameters.
*   **(January 22, 2025):** sync\_tensorboardX & sync\_tensorboard\_torch.
*   **(January 17, 2025):** sync\_wandb, Improved Log Rendering.
*   **(January 11, 2025):** Project Table Performance, Dragging, Sorting, and Filtering.
*   **(January 01, 2025):** Persistent Smoothing, Chart Resizing.
*   **(December 22, 2024):** LLaMA Factory Integration.
*   **(December 15, 2024):** Hardware Monitoring (0.4.0).
*   **(December 06, 2024):** LightGBM and XGBoost Integration, Increased Log Length Limit.
*   **(November 26, 2024):** Huawei Ascend NPU, Kunpeng CPU, QingCloud Foundation Computing.

</details>

<br>

## Quickstart

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

<details><summary>Offline Dashboard Installation</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login & API Key

1.  [Register](https://swanlab.cn)
2.  Get your API Key ([API Key](https://swanlab.cn/settings))
3.  Login in the terminal

```bash
swanlab login
```

### 3. Integrate SwanLab into your code

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

Go to [SwanLab](https://swanlab.cn) and check your experiment!

<br>

## Self-Hosting

SwanLab's community edition supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted Version with Docker

Refer to the documentation: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For China, install:

```bash
./install.sh
```

For DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point Experiments to Self-Hosted

Login to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

<br>

## Hardware Record

SwanLab tracks hardware information and resource usage during AI training.

| Hardware | Info Record | Resource Monitoring | Script                                                              |
| :------- | :---------- | :------------------ | :------------------------------------------------------------------ |
| NVIDIA GPU     | ✅         | ✅                   | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)    |
| 昇腾NPU    | ✅         | ✅                   | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)    |
| 苹果SOC   | ✅         | ✅                   | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)    |
| 寒武纪MLU | ✅         | ✅                   | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| 昆仑芯XPU  | ✅         | ✅                   | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| 摩尔线程GPU  | ✅         | ✅                   | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| 沐曦GPU  | ✅         | ✅                   | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| 海光DCU  | ✅         | ✅                   | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU         | ✅         | ✅                   | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)        |
| 内存       | ✅         | ✅                   | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)     |
| 硬盘        | ✅         | ✅                   | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)         |
| 网络 | ✅         | ✅                   | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)       |

<br>

## Framework Integrations

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)
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
*   [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)
*   [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
*   [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
*   [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
*   [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
*   [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)
*   [Stable Baseline3](https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html)
*   [veRL](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)
*   [HuggingFace trl](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-trl.html)
*   [EasyR1](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
*   [AReaL](https://docs.swanlab.cn/guide_cloud/integration/integration-areal.html)
*   [ROLL](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)
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

## Plugins and API

Enhance your experiment management with SwanLab plugins!

*   [Custom Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notification](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notification](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [Dingtalk Notification](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeCom Notification](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notification](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notification](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [Filelogdir Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparison

SwanLab is designed to be a modern, user-friendly alternative to existing experiment tracking tools. Here's how it stacks up:

### TensorBoard vs. SwanLab

*   **Cloud Support**: Easily sync and save your training experiments online, allowing remote monitoring, project management, experiment sharing, real-time notifications, and multi-device access. TensorBoard is a local tracking tool.
*   **Team Collaboration**:  Manage training projects effectively, share experiment links, and facilitate team discussions. TensorBoard is primarily for individual use.
*   **Persistent Dashboard**:  Access your results in a centralized dashboard, regardless of where you train.  TensorBoard requires manual TFEvent file management.
*   **Superior Table Functionality**:  Search, filter, and compare results across many experiments. TensorBoard is not designed for large projects.

### Weights and Biases vs SwanLab

*   Weights & Biases is a closed-source MLOps platform that requires an internet connection.
*   SwanLab supports online and self-hosted versions.

<br>

## Community

### Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official Documentation
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline Dashboard Web Code
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-Hosting Scripts

### Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report issues.
*   [Email Support](zeyi.lin@swanhub.co): Contact us with questions.
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discuss issues.

### SwanLab README Badges

Use the SwanLab badge in your README:

[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

Find design assets: [assets](https://github.com/SwanHubX/assets)

### Cite SwanLab

If SwanLab helped your research, please use this format:

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

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for information.

Support SwanLab by sharing on social media, events, and conferences!

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