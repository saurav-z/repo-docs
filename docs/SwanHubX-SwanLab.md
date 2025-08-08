<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: Open-Source AI Experiment Tracking & Visualization

**SwanLab empowers AI researchers to track, visualize, and collaborate on their machine learning experiments with ease.** ([Original Repo](https://github.com/SwanHubX/SwanLab))

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Tracking SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)


![](readme_files/swanlab-overview.png)

[中文 / English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features of SwanLab

*   **Experiment Tracking & Visualization:**
    *   Visualize training metrics in real-time with interactive charts (line charts, media, 3D point clouds, and more).
    *   Track hyperparameters, model configurations, and system metrics.
    *   Supports various data types: scalar metrics, images, audio, text, video, 3D point clouds, biochemical molecules, and custom ECharts charts.
    *   [LLM content visualization](https://docs.swanlab.cn/guide_cloud/experiment_track/log-text.html) and charts.

    ![swanlab-table](readme_files/molecule.gif)
    ![swanlab-echarts](readme_files/echarts.png)
    ![text-chart](readme_files/text-chart.gif)

*   **Framework Integrations:** Seamlessly integrate with 30+ popular machine learning frameworks, including:
    *   PyTorch, HuggingFace Transformers, PyTorch Lightning, LLaMA Factory, MMDetection, Ultralytics, PaddleDetetion, LightGBM, XGBoost and more.
    *   Check [integrations](https://docs.swanlab.cn/guide_cloud/integration/) for the list.

    ![](readme_files/integrations.png)

*   **Hardware Monitoring:**
    *   Real-time monitoring of CPU, GPU (Nvidia, Ascend, Apple SOC, Cambricon, Kunlunxin, Moore Threads, Metax, Hygon), memory, disk, and network usage.

*   **Experiment Management:**
    *   Centralized dashboard for managing projects and experiments.
    *   Compare experiments side-by-side with tables and charts.
    *   Organize experiments with tags and grouping.

    ![](readme_files/swanlab-table.png)

*   **Collaboration & Sharing:**
    *   Collaborative training environment for teams.
    *   Share experiments with persistent URLs.

*   **Self-Hosting Support:** Use SwanLab locally or on your own servers.

*   **Plugin Extensibility:** Extend SwanLab's functionality with custom plugins and expand features, such as [Slack notifications](https://docs.swanlab.cn/plugin/notification-slack.html).

> \[!IMPORTANT]
>
> **Star the project** to stay updated on releases! ⭐️

![star-us](readme_files/star-us.png)

<br>

## Getting Started

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source Code</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
cd SwanLab
pip install -e .
```
</details>

<details><summary>Install Offline Dashboard Extension</summary>

```bash
pip install 'swanlab[dashboard]'
```
</details>


### 2. Login and Get Your API Key

1.  [Register](https://swanlab.cn) for a free account.

2.  Login to your account and copy the API key from User Settings > [API Key](https://swanlab.cn/settings).

3.  Open your terminal and run:

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

Go to [SwanLab](https://swanlab.cn) to view your first experiment.

<br>

## Use Cases & Examples

*   **Demo projects**
    *   [ResNet50 Cat/Dog Classification][demo-cats-dogs]
    *   [Yolov8-COCO128 Object Detection][demo-yolo]
    *   [Qwen2 Instruction Fine-tuning][demo-qwen2-sft]
    *   [LSTM Google Stock Prediction][demo-google-stock]
    *   [ResNeXt101 Audio Classification][demo-audio-classification]
    *   [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl]
    *   [EasyR1 Multi-Modal LLM RL Training][demo-easyr1-rl]
    *   [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo]

    [More examples](https://docs.swanlab.cn/zh/examples/mnist.html)

*   **Great Tutorials**
    *   [happy-llm](https://github.com/datawhalechina/happy-llm)
    *   [self-llm](https://github.com/datawhalechina/self-llm)
    *   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek)
*   **Referenced in papers**
    *   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
    *   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
    *   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)

🌟 Submit a PR with your tutorial if you would like to be included!

<br>

## Self-Hosting Guide

SwanLab supports self-hosting to view the dashboard offline.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy the Self-Hosted Version with Docker

For detailed instructions, refer to the [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Quick installation for China region:

```bash
./install.sh
```

Install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point Experiments to Your Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, experiments will be recorded to your self-hosted service.

<br>

## Hardware Monitoring

SwanLab records hardware information and resource usage during AI training.

| Hardware | Information Recorded | Resource Monitoring | Script |
| --- | --- | --- | --- |
| Nvidia GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU | ✅ | ✅ | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC | ✅ | ✅ | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU | ✅ | ✅ | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅ | ✅ | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU     | ✅        | ✅        | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory        | ✅        | ✅        | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk        | ✅        | ✅        | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Submit an Issue or PR if you wish to record other hardware.

<br>

## Integrations

Combine SwanLab with your favorite frameworks.

### Frameworks

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

**Computer Vision Frameworks**

*   [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
*   [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
*   [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
*   [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
*   [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)

**Reinforcement Learning Frameworks**

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

## Plugins and API

Extend SwanLab's functionality.

*   [Create Your Own Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Lark Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeCom Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [FileLogDir Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open APIs:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparisons

### TensorBoard vs. SwanLab

*   **Cloud Support:** SwanLab allows you to sync and save your training experiments online for easy remote access, progress tracking, experiment management, sharing links, and multi-device viewing. TensorBoard is an offline experiment tracking tool.
*   **Collaboration:** SwanLab facilitates collaborative machine learning projects by managing training projects, sharing experiment links, and fostering discussions. TensorBoard is primarily designed for individual use.
*   **Persistent Dashboard:** SwanLab provides a consistent dashboard regardless of where your models are trained - local machines, lab clusters, or cloud GPUs. TensorBoard requires effort to manage TFEvent files.
*   **Advanced Tables:** SwanLab's tables allow you to view, search, and filter results from different experiments, making it easy to analyze thousands of model versions and find the best-performing models.  TensorBoard is not well-suited for large projects.

### Weights and Biases vs. SwanLab

*   Weights and Biases is a closed-source MLOps platform that requires an internet connection.
*   SwanLab is open-source, free, and offers self-hosting options, providing both online and offline functionality.

<br>

## Community

### Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs) - Official Documentation
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard) - Offline Dashboard Code
*   [self-hosted](https://github.com/swanhubx/self-hosted) - Self-Hosting Scripts

### Community and Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues) - Report issues and ask questions.
*   [Email Support](zeyi.lin@swanhub.co) - Get support.
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a> - Discuss and share your experiences.

### SwanLab README Badges

Add SwanLab badges to your README:

[![Tracking SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Cite SwanLab

If you find SwanLab helpful for your research, please cite it:

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

Read the [contribution guide](CONTRIBUTING.md) to contribute.

We welcome support and shares through social media and events!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## License

SwanLab is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).