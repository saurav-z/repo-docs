<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

## SwanLab: Unleash the Power of Visualized AI Experiment Tracking 🚀

SwanLab is an open-source, modern, and user-friendly tool for tracking, visualizing, and collaborating on your deep learning experiments, making it easier to understand, debug, and share your model training progress.  <a href="https://github.com/SwanHubX/SwanLab">Explore the code on GitHub!</a>

[🔥SwanLab Online](https://swanlab.cn) | [📃 Documentation](https://docs.swanlab.cn) | [🤔 Issues](https://github.com/swanhubx/swanlab/issues) | [💬 Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) | [📰 Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) | [🤝 Community](https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg) | [📊 Benchmarks](https://swanlab.cn/benchmarks)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![Wechat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

![](readme_files/swanlab-overview.png)

中文 / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>

<br/>

## Key Features

*   **Experiment Tracking and Visualization**: Track and visualize key metrics, hyperparameters, and model outputs in real-time, enabling you to understand your model's performance, diagnose issues, and iterate quickly.
    *   **Versatile Data Types**: Supports a wide array of data types, including scalar metrics, images, audio, text, videos, 3D point clouds, biochemical molecules, and custom ECharts charts.
    *   **Rich Charting Options**: Offers a variety of chart types like line charts, media charts (images, audio, text, video), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, and custom charts.
    *   **LLM-Specific Visualization**: Includes specialized text content visualization charts designed for large language model training, with Markdown rendering support.
    *   **Background Auto-Recording**: Automatically logs hardware environment, Git repository information, Python environment details, and project directory information.
    *   **Resume Training Support**: Allows you to add new metric data to the same experiment after training completion or interruption.
*   **Comprehensive Framework Integrations**: Seamlessly integrates with over 30 popular machine learning frameworks, including PyTorch, TensorFlow, Hugging Face Transformers, PyTorch Lightning, LLaMA Factory, and many more.
*   **Hardware Monitoring**: Provides real-time monitoring and recording of system-level hardware metrics such as CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (Mxic), Moore Threads GPU, memory, disk usage, and network activity.
*   **Experiment Management**: Offers a centralized dashboard designed for training scenarios, allowing you to quickly manage multiple projects and experiments with a global overview.
*   **Result Comparison**: Enables easy comparison of hyperparameter and results across different experiments through online tables and comparison charts, facilitating the discovery of iteration insights.
*   **Collaborative Experimentation**: Supports collaborative training, allowing real-time experiment synchronization within a project, enabling you to view team training records online and share insights.
*   **Shareable Results**: Provides shareable experiment URLs to easily share individual experiment results with colleagues or embed them in online notes.
*   **Self-Hosting Support**: Supports offline usage and self-hosting, enabling you to view dashboards and manage experiments in a self-hosted environment.
*   **Plugin Extensibility**: Allows you to extend SwanLab's capabilities through plugins, enhancing your experiment management experience, with available plugins for [Feishu notifications](https://docs.swanlab.cn/plugin/notification-lark.html), [Slack notifications](https://docs.swanlab.cn/plugin/notification-slack.html), and [CSV recorders](https://docs.swanlab.cn/plugin/writer-csv.html), among others.

> \[!IMPORTANT]
>
> **Star the project** to stay informed about all releases and updates! ⭐️

![star-us](readme_files/star-us.png)

<br>

## Getting Started

### Installation

```bash
pip install swanlab
```

*   Install from source:

```bash
git clone https://github.com/SwanHubX/SwanLab.git
cd SwanLab
pip install -e .
```

*   Install offline dashboard extensions:

```bash
pip install 'swanlab[dashboard]'
```

### Usage

1.  **Sign up** for a free account on [SwanLab](https://swanlab.cn) and get your API key.

2.  **Login** in the terminal

    ```bash
    swanlab login
    ```

    Enter your API key when prompted.

3.  **Integrate SwanLab into your code**

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

  See your experiment results on [SwanLab](https://swanlab.cn)!

<br>

## Examples & Tutorials

*   **MNIST Hand-written Digits Recognition** - [link](https://docs.swanlab.cn/examples/mnist.html)
*   **FashionMNIST Clothing Classification** - [link](https://docs.swanlab.cn/examples/fashionmnist.html)
*   **CIFAR10 Image Classification** - [link](https://docs.swanlab.cn/examples/cifar10.html)
*   **Resnet Cat and Dog Classification** - [link](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   **YOLO Object Detection** - [link](https://docs.swanlab.cn/examples/yolo.html)
*   **UNet Medical Image Segmentation** - [link](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   **Audio Classification** - [link](https://docs.swanlab.cn/examples/audio_classification.html)
*   **DQN Reinforcement Learning - CartPole** - [link](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   **LSTM Google Stock Prediction** - [link](https://docs.swanlab.cn/examples/lstm.html)
*   **BERT Text Classification** - [link](https://docs.swanlab.cn/examples/bert.html)
*   **Stable Diffusion Text-to-Image Fine-tuning** - [link](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   **LLM Pretraining** - [link](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   **GLM4 Instruction Fine-tuning** - [link](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   **Qwen Downstream Task Training** - [link](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   **NER Named Entity Recognition** - [link](https://docs.swanlab.cn/examples/ner.html)
*   **Qwen3 Medical Model Fine-tuning** - [link](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   **Qwen2-VL Multimodal LLM Fine-tuning** - [link](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   **GRPO LLM Reinforcement Learning** - [link](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   **Qwen3-SmVL-0.6B Multimodal Model Training** - [link](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   **LeRobot Embodied AI Tutorial** - [link](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)

🌟 We welcome PRs for your tutorials!

<br>

## Self-Hosting

Self-hosted community version supports offline view of SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### Deploy the self-hosted version using Docker

For details, see: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker

# China Mainland Installation
./install.sh

# Pull from DockerHub to install
./install-dockerhub.sh
```

### Point experiments to your self-hosted service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Once logged in, experiment records will be directed to the self-hosted service.

<br>

## Hardware Monitoring

SwanLab records and monitors hardware information and resource usage during AI training. Below is the support table:

| Hardware          | Info Record | Resource Monitoring | Script                                                                    |
| ----------------- | ----------- | ------------------- | ------------------------------------------------------------------------- |
| Nvidia GPU        | ✅          | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU        | ✅          | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC         | ✅          | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU     | ✅          | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU     | ✅          | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅          | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU         | ✅          | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU         | ✅          | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU               | ✅          | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)         |
| Memory            | ✅          | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)    |
| Disk              | ✅          | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)        |
| Network           | ✅          | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

If you want to record other hardware, welcome to submit Issues and PRs!

<br>

## Framework Integrations

Combine your favorite framework with SwanLab!

Below is a list of frameworks we have integrated. Welcome to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to provide feedback on the frameworks you want to integrate.

**Base Frameworks**

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialty/Fine-tuning Frameworks**

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

## Plugins and APIs

Extend SwanLab functionality with plugins to enhance your experiment management!

*   [Customize your plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notification](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notification](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notification](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Work Notification](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notification](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notification](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Recorder](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Directory Recorder](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open APIs:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparison with Similar Tools

### TensorBoard vs. SwanLab

*   **Cloud Support**: SwanLab provides convenient cloud synchronization and experiment saving, suitable for remote training progress monitoring, project management, experiment sharing, real-time notification, and multi-device access, unlike TensorBoard.
*   **Collaboration**: Facilitates collaboration with others via SwanLab for efficient project management, experiment sharing, and communication in machine learning projects. TensorBoard is primarily designed for personal use.
*   **Centralized Dashboard**: Records your results to a centralized dashboard whether you train models on a local machine, in a lab cluster, or on public cloud GPU instances. You need to spend time copying and managing TFEvent files with TensorBoard.
*   **Powerful Table**: View, search, and filter results from different experiments within SwanLab, making it easy to view thousands of model versions and to find the best-performing models for various tasks. TensorBoard is not suitable for large projects.

### Weights & Biases vs. SwanLab

*   Weights & Biases is a closed-source MLOps platform requiring an internet connection.
*   SwanLab supports online use but also offers open-source, free, self-hosted versions.

<br>

## Community

### Repositories

*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting scripts repository
*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard repository, which holds the web code of the lightweight offline dashboard opened by `swanlab watch`.

### Community and Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): For errors and issues you encounter using SwanLab
*   [Email Support](zeyi.lin@swanhub.co): To report problems and concerns about using SwanLab
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discuss and share issues using SwanLab and the latest AI technologies

### SwanLab README Badges

If you like using SwanLab in your work, add the SwanLab badge to your README:

[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design materials: [assets](https://github.com/SwanHubX/assets)

### Cite SwanLab in your Paper

If you find SwanLab helpful in your research, please consider citing it in the following format:

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

Interested in contributing to SwanLab? Start by reading the [contribution guide](CONTRIBUTING.md).

We welcome support for SwanLab through social media, events, and conferences! Thank you!

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