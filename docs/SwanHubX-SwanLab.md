<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: Supercharge Your AI Training with Open-Source Experiment Tracking

**SwanLab is an open-source, user-friendly AI model training tracker and visualizer, providing a comprehensive platform for tracking, logging, comparing, and collaborating on your machine learning experiments.**

[Visit the SwanLab GitHub Repository](https://github.com/SwanHubX/SwanLab) for more details and to contribute.

<div align="center">
  <a href="https://swanlab.cn">🔥SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Documentation</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Suggest Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Baseline Community</a>

  [![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
  [![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
  [![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
  [![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
  [![GitHub Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
  [![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
  [![SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
  [![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
  [![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
  [![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
  [![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

  ![](readme_files/swanlab-overview.png)

  <a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</div>

<br/>

## Key Features

*   **Experiment Tracking & Visualization:** Visualize metrics, hyperparameters, and model outputs in real-time with interactive charts and dashboards.
*   **Automatic Logging:** Automatically log key data, including model parameters, hardware usage, and code changes.
*   **Flexible Framework Integrations:** Seamlessly integrate with popular deep learning frameworks, including PyTorch, TensorFlow, and more.
*   **Hardware Monitoring:** Monitor CPU, GPU, memory, and other hardware resource usage during training.
*   **Experiment Comparison:** Compare and contrast experiments side-by-side to identify the best-performing models.
*   **Team Collaboration:** Share experiments and collaborate with your team using SwanLab's online features.
*   **Self-Hosted Option:** Run SwanLab locally or on your own server for complete data control.
*   **Customizable and Extensible:** Extend SwanLab's functionality with plugins for notifications, custom visualizations, and more.

<br/>

## Core Capabilities:

**1. 📊 Comprehensive Experiment Tracking:** Easily track key metrics, hyperparameters, and model outputs.

*   **Cloud Support:** Utilize the cloud version for anytime, anywhere access.
*   **Parameter Logging:** Record crucial details about your model's configuration.
*   **Metric Summarization:** Automatically summarize key performance indicators.
*   **Table Analysis:** Analyze and compare results in tabular form.
*   **Rich Visualization:** Visualize training progress with intuitive UI.
*   **Supported Metadata Types:** Scalar metrics, images, audio, text, video, 3D point clouds, biochemical molecules, and custom Echarts charts.

![swanlab-table](readme_files/molecule.gif)
![swanlab-echarts](readme_files/echarts.png)
![text-chart](readme_files/text-chart.gif)

*   **Automatic Background Logging**: Capture logs, hardware metrics, git details, Python environment, and library lists.
*   **Checkpoint Resuming**: Resume training and add new data to existing experiments seamlessly.

**2. ⚡️ Wide Range of Framework Integrations:** Compatible with 30+ frameworks:

![](readme_files/integrations.png)

**3. 💻 Hardware Monitoring:** Monitor CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (Mxic), Moore Threads GPU (MThreads), memory

**4. 📦 Experiment Management:** Manage multiple projects and experiments with an intuitive dashboard.

**5. 🆚 Result Comparison:** Compare hyperparameters and results across experiments.
![](readme_files/swanlab-table.png)

**6. 👥 Online Collaboration:** Collaborate with your team on training, and visualize all your team's data.

**7. ✉️ Easy Sharing:** Share each experiment with a unique URL.

**8. 💻 Self-Hosting:** Use the self-hosted community version to see dashboards.

**9. 🔌 Plugin Ecosystem:** Expand the features with plugins, like  [Feishu notification](https://docs.swanlab.cn/plugin/notification-lark.html), [Slack notification](https://docs.swanlab.cn/plugin/notification-slack.html), [CSV logger](https://docs.swanlab.cn/plugin/writer-csv.html).

> \[!IMPORTANT]
>
> **Star the project** to be updated instantly! ⭐️

![star-us](readme_files/star-us.png)

<br>

## 📃 Online Demos

| [ResNet50 猫狗分类][demo-cats-dogs] | [Yolov8-COCO128 目标检测][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Tracks a simple ResNet50 model training on a cats and dogs image classification dataset. | Tracks training hyperparameters and metrics using Yolov8 in the COCO128 dataset on a target detection task. |

| [Qwen2 指令微调][demo-qwen2-sft] | [LSTM Google 股票预测][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Tracks Qwen2 LLM SFT training. | Predicts the price of Google stock using a basic LSTM model. |

| [ResNeXt101 音频分类][demo-audio-classification] | [Qwen2-VL COCO数据集微调][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiment progress from ResNet to ResNeXt on audio classification tasks. | Lora finetuning on the COCO2014 dataset based on Qwen2-VL multi-modal large model. |

| [EasyR1 多模态LLM RL训练][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO训练][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Training multimodal LLM RL using the EasyR1 framework | Training GRPO using Qwen2.5-0.5B model on the GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Code Installation</summary>

For the latest features, install from source.

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Installation</summary>

[Offline dashboard documentation](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get API Key

1.  [Register a free account](https://swanlab.cn)
2.  Log in, and in User Settings > [API Key](https://swanlab.cn/settings) copy your API Key.
3.  In the terminal, enter:

```bash
swanlab login
```

Enter your API Key when prompted.

### 3. Integrate SwanLab into Your Code

```python
import swanlab

# Initialize a new swanlab experiment
swanlab.init(
    project="my-first-ml",
    config={'learning-rate': 0.003},
)

# Record metrics
for i in range(10):
    swanlab.log({"loss": i, "acc": i})
```

<br>

## 💻 Self-Hosting

Self-hosted community version supports offline viewing of SwanLab dashboards.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted Version with Docker

See: [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For China:

```bash
./install.sh
```

For DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point Experiments to Self-Hosted Service

Login to your service:

```bash
swanlab login --host http://localhost:8000
```

<br>

## 🔥 Real-World Examples

**Projects using SwanLab:**
- [happy-llm](https://github.com/datawhalechina/happy-llm): Beginner-friendly LLM tutorial ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
- [self-llm](https://github.com/datawhalechina/self-llm): Guide on tuning LLMs, including open-source models ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
- [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): Tutorials and research on the DeepSeek series ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
- [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): SmolVLM2's vision head with Qwen3-0.6B model ![GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)
- [OPPO/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models): An end-to-end agent foundation model through multi-agent distillation and agent RL.  ![GitHub Repo stars](https://img.shields.io/github/stars/OPPO-PersonalAI/Agent_Foundation_Models)

**Papers utilizing SwanLab:**
- [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
- [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
- [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
- [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorials:**
- [MNIST Handwriting Recognition](https://docs.swanlab.cn/examples/mnist.html)
- [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
- [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
- [Resnet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
- [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
- [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
- [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
- [DQN Reinforcement Learning - Cartpole Inverted Pendulum](https://docs.swanlab.cn/examples/dqn_cartpole.html)
- [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
- [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
- [Stable Diffusion Image Generation Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
- [LLM Pre-training](https://docs.swanlab.cn/examples/pretrain_llm.html)
- [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
- [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
- [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
- [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
- [Qwen2-VL Fine-tuning Tutorial](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
- [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
- [Qwen3-SmVL-0.6B Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
- [LeRobot Embodied Intelligence Guide](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
- [GLM-4.5-Air-LoRA and SwanLab visualization](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
- [How to RAG? SwanLab Document Assistant Released](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟 Submit a PR to add your tutorial!

<br>

## 🎮 Hardware Recording

SwanLab records the hardware information and resource usage during AI training.

| Hardware | Information Recording | Resource Monitoring | Script |
| --- | --- | --- | --- |
| NVIDIA GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU | ✅ | ✅ | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC | ✅ | ✅ | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU | ✅ | ✅ | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅ | ✅ | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Mxic GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU     | ✅        | ✅        | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory        | ✅        | ✅        | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk        | ✅        | ✅        | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

<br>

## 🚗 Framework Integrations

Integrate SwanLab with your favorite framework!  
List of frameworks that we've integrated, welcome to submit [Issue](https://github.com/swanhubx/swanlab/issues) to give us feedback.

**Basic Frameworks**
-   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
-   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
-   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Fine-tuning Frameworks**
-   [PyTorch Lightning](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-lightning.html)
-   [HuggingFace Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-transformers.html)
-   [LLaMA Factory](https://docs.swanlab.cn/guide_cloud/integration/integration-llama-factory.html)
-   [Modelscope Swift](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html)
-   [DiffSynth Studio](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html)
-   [Sentence Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-sentence-transformers.html)
-   [PaddleNLP](https://docs.swanlab.cn/guide_cloud/integration/integration-paddlenlp.html)
-   [OpenMind](https://modelers.cn/docs/zh/openmind-library/1.0.0/basic_tutorial/finetune/finetune_pt.html#%E8%AE%AD%E7%BB%83%E7%9B%91%E6%8E%A7)
-   [Torchtune](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-torchtune.html)
-   [XTuner](https://docs.swanlab.cn/guide_cloud/integration/integration-xtuner.html)
-   [MMEngine](https://docs.swanlab.cn/guide_cloud/integration/integration-mmengine.html)
-   [FastAI](https://docs.swanlab.cn/guide_cloud/integration/integration-fastai.html)
-   [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html)
-   [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html)
-   [MLX-LM](https://docs.swanlab.cn/guide_cloud/integration/integration-mlx-lm.html)

**Evaluation Frameworks**
- [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)

**Computer Vision**
-   [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
-   [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
-   [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
-   [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
-   [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)

**Reinforcement Learning**
-   [Stable Baseline3](https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html)
-   [veRL](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)
-   [HuggingFace trl](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-trl.html)
-   [EasyR1](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
-   [AReaL](https://docs.swanlab.cn/guide_cloud/integration/integration-areal.html)
-   [ROLL](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

**Other Frameworks:**
-   [Tensorboard](https://docs.swanlab.cn/guide_cloud/integration/integration-tensorboard.html)
-   [Weights&Biases](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html)
-   [MLFlow](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html)
-   [HuggingFace Accelerate](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html)
-   [Ray](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html)
-   [Unsloth](https://docs.swanlab.cn/guide_cloud/integration/integration-unsloth.html)
-   [Hydra](https://docs.swanlab.cn/guide_cloud/integration/integration-hydra.html)
-   [Omegaconf](https://docs.swanlab.cn/guide_cloud/integration/integration-omegaconf.html)
-   [OpenAI](https://docs.swanlab.cn/guide_cloud/integration/integration-openai.html)
-   [ZhipuAI](https://docs.swanlab.cn/guide_cloud/integration/integration-zhipuai.html)

[More Integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br>

## 🔌 Plugins and API

Extend SwanLab's functionality.

- [Customize Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
- [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
- [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
- [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
- [WeCom Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
- [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
- [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
- [CSV Logger](https://docs.swanlab.cn/plugin/writer-csv.html)
- [File Logger](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

API:
- [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparisons

### TensorBoard vs. SwanLab

*   **Cloud Support**: SwanLab enables cloud-based experiment syncing and data storage, for remote progress monitoring, project management, experiment sharing, real-time notifications, and multi-device access. TensorBoard is a local tool.

*   **Multi-user Collaboration:** SwanLab supports collaborative project management, experiment sharing, and communication. TensorBoard is designed for personal use.

*   **Persistent, Centralized Dashboard:** Results are recorded in one dashboard no matter where you train. In TensorBoard, you must spend time copying and managing files.

*   **Enhanced Tables:** SwanLab tables allow you to browse, search, and filter, making it easy to see different experiments and find the best-performing model for different tasks. TensorBoard does not support large projects.

### Weights and Biases vs. SwanLab

*   Weights and Biases is a proprietary MLOps platform requiring internet access.

*   SwanLab is open-source, free, and offers a self-hosted version.

<br>

## 👥 Community

### Related Repositories

-   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting deployment scripts
-   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation repository
-   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Lightweight offline dashboard for `swanlab watch`

### Community and Support

-   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues)
-   [Email Support](zeyi.lin@swanhub.co)
-   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>

### SwanLab README Badges

Add a SwanLab badge to your README:

[![SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Publications

If SwanLab has been useful for your research, please cite it:

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

Read [CONTRIBUTING.md](CONTRIBUTING.md).

We appreciate shares through social media.

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## 📃 License

SwanLab is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)