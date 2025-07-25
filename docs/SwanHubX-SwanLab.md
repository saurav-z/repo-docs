<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br/>

## SwanLab: Effortlessly Track, Visualize, and Collaborate on Your Deep Learning Experiments

SwanLab is an open-source, user-friendly tool designed to revolutionize your deep learning workflow by providing comprehensive experiment tracking, visualization, and collaboration features, seamlessly integrating with 30+ popular frameworks.  

<a href="https://swanlab.cn">🔥SwanLab 在线版</a> · <a href="https://docs.swanlab.cn">📃 文档</a> · <a href="https://github.com/swanhubx/swanlab/issues">报告问题</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">建议反馈</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">更新日志</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">基线社区</a>

[![][release-shield]][release-link]
[![][dockerhub-shield]][dockerhub-link]
[![][github-stars-shield]][github-stars-link]
[![][github-issues-shield]][github-issues-shield-link]
[![][github-contributors-shield]][github-contributors-link]
[![][license-shield]][license-shield-link]  
[![][tracking-swanlab-shield]][tracking-swanlab-shield-link]
[![][last-commit-shield]][last-commit-shield-link]
[![][pypi-version-shield]][pypi-version-shield-link]
[![][wechat-shield]][wechat-shield-link]
[![][pypi-downloads-shield]][pypi-downloads-shield-link]
[![][colab-shield]][colab-shield-link]

![](readme_files/swanlab-overview.png)

中文 / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 加入我们的[微信群](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features of SwanLab:

*   **Experiment Tracking & Visualization**:
    *   Track and visualize key metrics, hyperparameters, and experiment details.
    *   Visualize training progress with an intuitive UI.
    *   Supports various data types: scalars, images, audio, text, video, 3D point clouds, molecule data, and custom ECharts charts.
    *   A great visualization tool for LLM research!
        *   Text chart: A dynamic tool that can render Markdown content.
        *   Auto log: Log logging data such as logs, hardware environments, Git repositories, and Python libraries.
        *   Resume training: Supports the supplement of data in the same experiment after training or interruption.
    *   **Custom Chart Type**: You can use SwanLab with Echarts, providing more than 20 chart types such as bar charts, pie charts, and histograms.

    *   **Examples of metrics and hyperparameter tracking**: Easy code integration allows you to track key metrics.
    *   **Online Cloud Services**: The training progress can be tracked at any time through the cloud (similar to Weights & Biases). [Mobile app viewing](https://docs.swanlab.cn/guide_cloud/general/app.html)
    *   **Metadata Support**: Scalar metrics, images, audios, texts, videos, 3D point clouds, biochemical molecules, Echarts charts are all supported.
    *   **Chart Types**: Supported chart types include line charts, media charts (images, audios, texts, videos), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, and [custom charts](https://docs.swanlab.cn/guide_cloud/experiment_track/log-custom-chart.html).
    *   **LLM Text Visualization Component**: The Text visualization chart designed for large language model training scenarios, supporting Markdown rendering.
    *   **Background Auto Recording**: Logging logging, hardware environment, Git repository, Python environment, Python library list, and project running directory.
    *   **Checkpoint Resume Recording**: Supports adding new metric data to the same experiment after training or interruption.

    ![swanlab-table](readme_files/molecule.gif)
    ![swanlab-echarts](readme_files/echarts.png)
    ![text-chart](readme_files/text-chart.gif)

*   **Extensive Framework Integration**: Seamlessly integrates with 30+ popular machine learning frameworks, including:  PyTorch, Hugging Face Transformers, PyTorch Lightning, LLaMA Factory, MMDetection, Ultralytics, PaddleDetetion, LightGBM, XGBoost, Keras, Tensorboard, Weights&Biases, OpenAI, Swift and more.
    ![integrations](readme_files/integrations.png)

*   **Hardware Monitoring**: Real-time monitoring of hardware resources, including CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (MX GPU), Moore Threads GPU, and memory usage.

*   **Experiment Management**:
    *   Centralized dashboard for managing projects and experiments.
    *   Quickly review the global overview through the overall view, and quickly manage multiple projects and experiments.

*   **Experiment Comparison**: Compare hyperparameters and results across different experiments with an online table and comparison chart to find inspiration for iteration.

    ![swanlab-table](readme_files/swanlab-table.png)

*   **Collaboration**: Collaborate on experiments with your team in real-time and view the training records of your team online, based on the results.

*   **Share Results**: Share experiments with a persistent URL for easy sharing or embedding in notes.

*   **Self-Hosting**: Supports offline and self-hosted use for local experiment viewing and management. [Self-Hosting Guide](#-自托管)

*   **Plugin Extensions**: Extend SwanLab's functionality with plugins, such as [Feishu notifications](https://docs.swanlab.cn/plugin/notification-lark.html), [Slack notifications](https://docs.swanlab.cn/plugin/notification-slack.html), and [CSV recorders](https://docs.swanlab.cn/plugin/writer-csv.html), etc.

> \[!IMPORTANT]
>
> **Star the project** to get all release notifications from GitHub without delay! ⭐️

![star-us](readme_files/star-us.png)

<br>

## Online Demos

Explore interactive SwanLab demos:

| [ResNet50 猫狗分类][demo-cats-dogs] | [Yolov8-COCO128 目标检测][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Tracks an image classification task trained on the cat and dog dataset using a simple ResNet50 model. | Uses Yolov8 to perform object detection tasks on the COCO128 dataset and tracks training hyperparameters and metrics. |

| [Qwen2 指令微调][demo-qwen2-sft] | [LSTM Google 股票预测][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Tracks Qwen2 large language model instruction fine-tuning training and completes simple instruction following. | Trains on the Google stock price dataset using a simple LSTM model to predict future stock prices. |

| [ResNeXt101 音频分类][demo-audio-classification] | [Qwen2-VL COCO数据集微调][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Gradual experiment process from ResNet to ResNeXt on audio classification tasks | Lora fine-tuning based on Qwen2-VL multi-modal large model on the COCO2014 dataset. |

| [EasyR1 多模态LLM RL训练][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO训练][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Use EasyR1 framework for multi-modal LLM RL training | GRPO training based on Qwen2.5-0.5B model on the GSM8k dataset |

[更多案例](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## Quick Start:

### 1. Installation:

```bash
pip install swanlab
```

<details><summary>Source Code Installation</summary>

If you want to experience the latest features, you can install from the source code.

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

### 2. Login and Get API Key:

1.  [Register an account](https://swanlab.cn) for free.

2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).

3.  Open your terminal and type:

```bash
swanlab login
```

When prompted, enter your API Key and press Enter to complete the login.

### 3. Integrate SwanLab into Your Code:

```python
import swanlab

# Initialize a new SwanLab experiment
swanlab.init(
    project="my-first-ml",
    config={'learning-rate': 0.003},
)

# Record metrics
for i in range(10):
    swanlab.log({"loss": i, "acc": i})
```

Congratulations! Go to [SwanLab](https://swanlab.cn) to view your first experiment.

<br>

## Self-Hosting:

The self-hosted community version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted Version Using Docker

See the documentation for details: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Fast installation in China:

```bash
./install.sh
```

Install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Specify Experiments to Self-Hosted Services

Log in to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Once logged in, you can record your experiments to the self-hosted service.

<br>

## Practical Examples

**Open Source Projects with Excellent Tutorials Using SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): Tutorial on the principles and practice of large language models from scratch ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm): "Open Source Large Model User Guide" tailored for Chinese users, based on Linux environments for quick fine-tuning (full parameters/Lora), deployment of domestic and foreign open-source models (LLM) / multi-modal large models (MLLM) tutorial ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series of work interpretation, expansion, and reproduction. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)

**Excellent Papers Using SwanLab:**

*   [Animation Needs Attention](https://arxiv.org/abs/2507.03916)

**Tutorial Articles:**

*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat and Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cartpole Inverted Pendulum](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multi-Modal Large Model Fine-tuning Practical](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multi-Modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)

<br>

## Hardware Monitoring:

SwanLab records **hardware information** and **resource usage** during AI training.  Here's a table of supported hardware:

| Hardware        | Information Recording | Resource Monitoring | Script                                                                  |
| --------------- | --------------------- | ------------------- | ----------------------------------------------------------------------- |
| Nvidia GPU      | ✅                    | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU      | ✅                    | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC       | ✅                    | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU   | ✅                    | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU   | ✅                    | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅                    | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MX GPU          | ✅                    | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU       | ✅                    | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU             | ✅                    | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)          |
| Memory          | ✅                    | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)      |
| Disk            | ✅                    | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)          |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)  |

If you want to record other hardware, please submit an issue and pull request!

<br>

## Framework Integrations:

Use SwanLab with your favorite framework!  
Here's a list of integrated frameworks; feel free to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to suggest the integration of a framework you want.

**Basic Frameworks**

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

<br>

## Plugins & API

Enhance your experiment management with plugins!

*   [Create your own plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Recorder](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Directory Recorder](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI:

*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparisons with Familiar Tools

### TensorBoard vs. SwanLab

*   **☁️ Online Support**: SwanLab makes it convenient to synchronize and store training experiments online through the cloud, making it easy to view training progress remotely, manage historical projects, share experiment links, send real-time message notifications, and view experiments from multiple terminals. Tensorboard is an offline experiment tracking tool.

*   **👥 Multi-Person Collaboration**:  SwanLab can easily manage training projects of multiple people and cross-team collaboration, share experiment links, and exchange ideas across spaces. Tensorboard is mainly designed for individuals and is difficult to carry out multi-person collaboration and experiment sharing.

*   **💻 Persistent, Centralized Dashboard**: Regardless of where you train your model, whether on a local computer, in a lab cluster, or in a public cloud GPU instance, your results are recorded in the same centralized dashboard. Using TensorBoard requires time to copy and manage TFEvent files from different machines.

*   **💪 More Powerful Tables**: Through SwanLab tables, you can view, search, and filter results from different experiments, and easily view thousands of model versions and find the best performance model for different tasks. TensorBoard is not suitable for large projects.

### Weights and Biases vs. SwanLab

*   Weights and Biases is a closed-source MLOps platform that must be used online.

*   SwanLab supports not only online use but also open-source, free, and self-hosted versions.

<br>

## Community:

### Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official Documentation Repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard repository, storing the web code of the lightweight offline dashboard opened by `swanlab watch`
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosted deployment script repository

### Community & Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Errors and problems encountered while using SwanLab
*   [Email Support](zeyi.lin@swanhub.co): Provide feedback on issues using SwanLab
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Discuss issues, and share the latest AI technology using SwanLab

### SwanLab README Badges

If you enjoy using SwanLab in your work, add a SwanLab badge to your README:

[![][tracking-swanlab-shield]][tracking-swanlab-shield-link]  [![][visualize-swanlab-shield]][visualize-swanlab-shield-link]

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design materials: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in your papers

If you find SwanLab helpful for your research, consider citing it with the following format:

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

Interested in contributing to SwanLab? First, take some time to read the [contribution guide](CONTRIBUTING.md).

We also warmly welcome supporting SwanLab through sharing on social media, events, and conferences! Thank you very much!

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

[github-issues-shield]: https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb
[github-issues-shield-link]: https://github.com/swanhubx/swanlab/issues

[github-contributors-shield]: https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square
[github-contributors-link]: https://github.com/swanhubx/swanlab/graphs/contributors

[demo-cats-dogs]: https://swanlab.cn/@ZeyiLin/Cats_Dogs_Classification/runs/jzo93k112f15pmx14vtxf/chart
[demo-cats-dogs-image]: readme_files/example-catsdogs.png

[demo-yolo]: https://swanlab.cn/@ZeyiLin/ultratest/runs/yux7vclmsmmsar9ear7u5/chart
[demo-yolo-image]: readme_files/example-yolo.png

[demo-qwen2-sft]: https://swanlab.cn/@ZeyiLin/Qwen2-f