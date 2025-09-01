<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

**SwanLab: Your Open-Source AI Training Companion for Tracking, Visualizing, and Collaborating on Deep Learning Experiments.**

[🔥SwanLab Online Version](https://swanlab.cn) · [📃 Documentation](https://docs.swanlab.cn) · [Report Issues](https://github.com/swanhubx/swanlab/issues) · [Suggest Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) · [Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> [Benchmarking Community](https://swanlab.cn/benchmarks)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)


![](readme_files/swanlab-overview.png)

[中文](README_CN.md) / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>


</div>

<br/>

## 🚀 Key Features of SwanLab

SwanLab empowers AI researchers with a powerful platform for managing and understanding their deep learning experiments. Here's what sets SwanLab apart:

*   **📊 Experiment Tracking & Visualization**:  Track crucial metrics, visualize training progress, and gain insights with intuitive charts and graphs. SwanLab supports various data types including scalar metrics, images, audio, text, video, 3D point clouds, and custom Echarts charts.

    *   **Comprehensive Metadata**: Log and analyze hyperparameters, metrics, and tabular data.
    *   **Rich Charting**:  Explore data with line charts, media views (images, audio, text, video), 3D point clouds, and custom Echarts charts.
    *   **LLM-Focused Visualization**: Dedicated text visualization for Large Language Model (LLM) training, including Markdown rendering.
    *   **Hardware Monitoring**: Real-time tracking of CPU, NPU (Ascend), GPU (Nvidia, etc.), and memory usage.
    *   **Automated Logging**:  Automatic capture of logs, hardware environment details, Git repository information, Python environment, and library lists.
    *   **Resume Training Support**:  Seamlessly continue logging metrics to existing experiments after interruptions.

*   **📦 Streamlined Experiment Management**: Organize and manage your experiments with a centralized dashboard.

*   **🆚 Comparative Analysis**: Compare and contrast experiments to identify trends and optimize models.

*   **🤝 Collaborative Workflows**:  Facilitate team collaboration by enabling real-time experiment synchronization, sharing results, and providing feedback.

*   **✉️ Effortless Sharing**:  Share experiment results with persistent URLs for easy collaboration and documentation.

*   **💻 Flexible Deployment**: Supports both cloud and offline usage with self-hosting options for added control and data privacy.

*   **🔌 Extensive Integrations**:  Seamlessly integrates with 30+ popular frameworks including PyTorch, Hugging Face Transformers, PyTorch Lightning, LLAMA Factory, and many more.

*   **🔌 Plugin Extensibility**:  Customize your workflow with plugins for notifications, CSV export, and more.

> \[!IMPORTANT]
>
> **Star the project** to receive all release notifications on GitHub! ⭐️

![star-us](readme_files/star-us.png)

<br/>

## 📃 Online Demos

Explore SwanLab's capabilities with these interactive demos:

| Demo                                              | Description                                  |
| ------------------------------------------------- | -------------------------------------------- |
| [ResNet50 Cats vs. Dogs Classification][demo-cats-dogs]   | Image classification using a simple ResNet50. |
| [Yolov8-COCO128 Object Detection][demo-yolo]        | Object detection with Yolov8.                |
| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft]    | Fine-tuning the Qwen2 large language model.    |
| [Google Stock Prediction][demo-google-stock]      | Time series prediction using LSTM.            |
| [ResNeXt101 Audio Classification][demo-audio-classification]| Audio classification with ResNeXt101. |
| [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] | Fine-tuning Qwen2-VL on the COCO2014 Dataset. |
| [EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl] | Training using EasyR1 framework.  |
| [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] | Training using GRPO.  |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br/>

## 🏁 Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Code Installation</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .
# OR
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Extension Installation</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get Your API Key

1.  [Register a free account](https://swanlab.cn)
2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings)
3.  Open your terminal and run:

```bash
swanlab login
```

Enter your API Key when prompted and press Enter.

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

Now, view your experiment results at [SwanLab](https://swanlab.cn).

<br/>

## 💻 Self-Hosting

Self-hosting provides a local dashboard for experiment visualization.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy with Docker

Refer to the [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For quick installation in China:

```bash
./install.sh
```

Or pull images from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point Experiments to Your Self-Hosted Service

Login:

```bash
swanlab login --host http://localhost:8000
```

<br/>

## 🔥 Practical Examples

Explore projects, papers, and tutorials utilizing SwanLab.

**Projects:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): LLM tutorials
*   [self-llm](https://github.com/datawhalechina/self-llm): LLM and MLLM tutorials
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek tutorials
*   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL):  Integrating SmolVLM2 and Qwen3.

**Papers:**

*   [Animation Needs Attention](https://arxiv.org/abs/2507.03916)
*   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
*   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
*   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorials:**

*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cats vs. Dogs Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/lstm-google-stock-prediction.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Finetuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Finetuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Finetuning](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 medical model Finetuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL multi-modal fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO LLM RL](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied AI](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA and SwanLab](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
*   [RAG](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟 Submit your tutorial contributions via a PR!

<br/>

## 🎮 Hardware Monitoring

Track hardware information and resource usage during AI training:

| Hardware | Information | Monitoring | Script |
| :------- | :---------- | :--------- | :----- |
| NVIDIA GPU  | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU  | ✅ | ✅ | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC | ✅ | ✅ | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU | ✅ | ✅ | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅ | ✅ | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU  | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU     | ✅        | ✅        | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory        | ✅        | ✅        | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk        | ✅        | ✅        | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network  | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Contribute with an Issue or PR if you need support for other hardware!

<br/>

## 🚗 Framework Integrations

Integrate SwanLab with your favorite frameworks: See [documentation](https://docs.swanlab.cn/guide_cloud/integration/).

**Base frameworks:**

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Fine-tuning frameworks:**

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

**Evaluation frameworks:**

*   [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)

**Computer Vision frameworks:**

*   [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
*   [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
*   [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
*   [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
*   [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)

**Reinforcement Learning frameworks:**

*   [Stable Baseline3](https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html)
*   [veRL](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)
*   [HuggingFace trl](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-trl.html)
*   [EasyR1](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
*   [AReaL](https://docs.swanlab.cn/guide_cloud/integration/integration-areal.html)
*   [ROLL](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

**Other frameworks:**

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

<br/>

## 🔌 Plugins & API

Extend SwanLab's functionality with plugins:

*   [Custom Plugin Guide](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Directory Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br/>

## 🆚 Comparison: SwanLab vs. Other Tools

### TensorBoard vs. SwanLab

*   **Cloud Support**: SwanLab offers cloud synchronization and management of training experiments for remote monitoring, management, sharing, and multi-device access. TensorBoard is an offline tool.
*   **Collaboration**: SwanLab is designed for multi-user, cross-team collaboration with project management, shared links, and communication features. TensorBoard primarily targets individual use.
*   **Centralized Dashboard**: Your results are recorded in a centralized dashboard, regardless of the training location. TensorBoard requires manual management of TFEvent files.
*   **Enhanced Tables**: SwanLab tables allow you to view, search, and filter results from various experiments, facilitating analysis of large projects.

### Weights & Biases vs. SwanLab

*   Weights & Biases is a proprietary, cloud-based MLOps platform.
*   SwanLab provides open-source, free, and self-hosted options.

<br/>

## 👥 Community

### Resources

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Documentation
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline Dashboard Code
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting scripts

### Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report issues
*   [Email Support](zeyi.lin@swanhub.co): Feedback on SwanLab
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Discuss SwanLab and AI

### SwanLab README Badges

Add a SwanLab badge to your README:

[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![SwanLab Visualization](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```markdown
[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![SwanLab Visualization](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab

Cite SwanLab in your research:

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

### Contribute to SwanLab

Read the [Contribution Guidelines](CONTRIBUTING.md).

We appreciate your support via social media, events, and conferences!

<br/>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br/>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## 📃 License

This project is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)