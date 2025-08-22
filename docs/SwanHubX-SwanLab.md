<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br/>

## 🚀 SwanLab: The Open-Source Training Tracker and Visualizer

**SwanLab is your all-in-one, open-source solution for tracking, visualizing, and collaborating on your deep learning experiments, providing a streamlined workflow for researchers.**

**Key Features:**

*   **📊 Intuitive Visualization:** Easily visualize your training metrics, hyperparameters, and model performance with dynamic charts.
    *   **Supported Data Types:** Scalar metrics, images, audio, text, videos, 3D point clouds, biochemical molecules, custom ECharts.
    *   **Chart Types:** Line charts, media (images, audio, text, video), 3D point clouds, biochemical molecules, bar charts, scatter plots, heatmaps, pie charts, radar charts, and more.
    *   **LLM-Specific Visualization:** Text content visualization chart tailored for large language model training.
    ![text-chart](readme_files/text-chart.gif)
*   **📝 Automated Logging:** Automatically record training progress, including metrics, hyperparameters, system information (hardware, environment), and Git commits.
*   **💻 Comprehensive Framework Integrations:** Seamlessly integrates with over 30 popular machine learning frameworks, including PyTorch, TensorFlow, Hugging Face Transformers, and more.
    ![](readme_files/integrations.png)
*   **☁️ Cloud and Offline Support:** Use SwanLab in the cloud (similar to Weights & Biases) or run it locally for complete control. ([See cloud usage instructions](https://docs.swanlab.cn/guide_cloud/general/app.html))
*   **🚀 Hardware Monitoring:** Monitor CPU, GPU (Nvidia, Ascend, Murex, Kunlunxin, 沐曦, 海光, Moore Threads), memory, disk, and network usage in real-time.
*   **🤝 Collaborative Features:** Share experiments, invite collaborators, and compare results to foster teamwork and accelerate research.
*   **📦 Experiment Management:** Organize and manage multiple projects and experiments with a centralized dashboard.
*   **🆚 Result Comparison:** Compare hyperparameters and results across experiments using interactive tables and charts.
    ![](readme_files/swanlab-table.png)
*   **🔌 Extensible with Plugins:** Extend SwanLab's functionality with plugins for notifications (Slack, Lark, Email), CSV logging, and more.
*   **💾 Self-Hosting Support:** Deploy a self-hosted community edition to view dashboards and manage experiments.  ([Self-Hosting Guide](#-自托管))

> \[!IMPORTANT]
>
> **Star the project** to stay updated with the latest releases and features! ⭐

![star-us](readme_files/star-us.png)

**Visit the [SwanLab GitHub Repository](https://github.com/SwanHubX/SwanLab) for more information and to get started.**

<br>

## 📃 在线演示

Explore interactive demos to experience SwanLab's capabilities:

| [ResNet50 猫狗分类][demo-cats-dogs] | [Yolov8-COCO128 目标检测][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| 跟踪一个简单的 ResNet50 模型在猫狗数据集上训练的图像分类任务。 | 使用 Yolov8 在 COCO128 数据集上进行目标检测任务，跟踪训练超参数和指标。 |

| [Qwen2 指令微调][demo-qwen2-sft] | [LSTM Google 股票预测][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| 跟踪 Qwen2 大语言模型的指令微调训练，完成简单的指令遵循。 | 使用简单的 LSTM 模型在 Google 股价数据集上训练，实现对未来股价的预测。 |

| [ResNeXt101 音频分类][demo-audio-classification] | [Qwen2-VL COCO数据集微调][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| 从ResNet到ResNeXt在音频分类任务上的渐进式实验过程 | 基于Qwen2-VL多模态大模型，在COCO2014数据集上进行Lora微调。 |

| [EasyR1 多模态LLM RL训练][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO训练][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| 使用EasyR1框架进行多模态LLM RL训练 | 基于Qwen2.5-0.5B模型在GSM8k数据集上进行GRPO训练 |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Extension Installation</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and get your API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Log in and find your API Key in User Settings > [API Key](https://swanlab.cn/settings).
3.  Open a terminal and run:

```bash
swanlab login
```

Enter your API Key when prompted.

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

Now, view your first experiment on [SwanLab](https://swanlab.cn).

<br>

## 💻 Self-Hosting

Self-hosting enables offline viewing of your SwanLab dashboards.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploying a self-hosted version with Docker

Refer to the documentation: [Docker Deployment Guide](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For quick installation in China:

```bash
./install.sh
```

Install by pulling the image from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Direct Experiments to Your Self-Hosted Instance

Log in to your self-hosted instance:

```bash
swanlab login --host http://localhost:8000
```

After logging in, your experiments will be recorded in the self-hosted service.

<br>

## 🔥 Real-World Examples

**Excellent tutorial open-source projects using SwanLab:**
- [happy-llm](https://github.com/datawhalechina/happy-llm)：从零开始的大语言模型原理与实践教程 ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
- [self-llm](https://github.com/datawhalechina/self-llm)：《开源大模型食用指南》针对中国宝宝量身打造的基于Linux环境快速微调（全参数/Lora）、部署国内外开源大模型（LLM）/多模态大模型（MLLM）教程 ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
- [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek)：DeepSeek 系列工作解读、扩展和复现 ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
- [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): 将SmolVLM2的视觉头与Qwen3-0.6B模型进行了拼接微调 ![GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)

**Excellent Papers using SwanLab:**
- [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
- [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
- [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
- [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorials:**
- [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
- [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
- [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
- [Resnet Cat and Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
- [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
- [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
- [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
- [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
- [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
- [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
- [Stable Diffusion Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
- [LLM Pre-training](https://docs.swanlab.cn/examples/pretrain_llm.html)
- [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
- [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
- [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
- [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
- [Qwen2-VL Multimodal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
- [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
- [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
- [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
- [GLM-4.5-Air-LoRA and SwanLab Visualization Recording](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)

🌟 If you have a tutorial to include, welcome to submit a PR!

<br>

## 🎮 Hardware Recording

SwanLab records the hardware information and resource utilization during AI training. Here's the support table:

| Hardware       | Information Recording | Resource Monitoring | Script                                                                |
| -------------- | --------------------- | ------------------- | --------------------------------------------------------------------- |
| Nvidia GPU     | ✅                   | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)      |
| Ascend NPU     | ✅                   | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)      |
| Apple SOC      | ✅                   | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)      |
| Cambricon MLU  | ✅                   | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU  | ✅                   | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py)  |
| Moore Threads GPU| ✅                   | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| 沐曦GPU| ✅                   | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU    | ✅                   | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py)  |
| CPU            | ✅                   | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)             |
| Memory         | ✅                   | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)          |
| Disk           | ✅                   | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)             |
| Network        | ✅                   | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)          |

If you want to record other hardware, please submit an Issue or PR!

<br>

## 🚗 Framework Integrations

Use your favorite frameworks with SwanLab!

Here's a list of integrated frameworks.  Submit an [Issue](https://github.com/swanhubx/swanlab/issues) to request integrations.

**Core Frameworks**

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

## 🔌 Plugins and API

Extend SwanLab's capabilities with plugins to enhance your experiment management!

*   [Custom Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Dir Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI:

*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparisons with Similar Tools

### Tensorboard vs SwanLab

*   **☁️ Online Support:** SwanLab allows you to easily sync and store training experiments online, making it easier to view training progress remotely, manage historical projects, share experiment links, and provide multi-platform viewing and experiment notifications. Tensorboard is an offline experiment tracking tool.

*   **👥 Multi-Person Collaboration:** When collaborating on machine learning projects with multiple people or across teams, SwanLab makes it easy to manage training projects, share experiment links, and facilitate communication. Tensorboard is designed mainly for individual use and is difficult for multi-person collaboration and experimental sharing.

*   **💻 Persistent, Centralized Dashboard:** No matter where you train your model—on a local machine, a lab cluster, or a public cloud GPU instance—your results are recorded on the same centralized dashboard. Using TensorBoard requires time to copy and manage TFEvent files from different machines.

*   **💪 More Powerful Tables:** SwanLab tables allow you to view, search, and filter results from different experiments, making it easy to view thousands of model versions and find the best-performing model for different tasks. TensorBoard is not suitable for large projects.

### Weights & Biases vs SwanLab

*   Weights & Biases is a closed-source MLOps platform requiring internet access.

*   SwanLab supports both internet access and the option for an open-source, free, and self-hosted version.

<br>

## 👥 Community

### Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official Documentation Repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline Dashboard Repository
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosted deployment scripts repository

### Community & Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): For errors and issues when using SwanLab
*   [Email Support](zeyi.lin@swanhub.co): For questions regarding the use of SwanLab
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): For discussing issues with SwanLab and sharing the latest AI techniques

### SwanLab README Badges

Add SwanLab badges to your README:

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design resources: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Publications

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

Interested in contributing to SwanLab? Please read the [Contribution Guide](CONTRIBUTING.md) first.

We greatly appreciate support through social media, events, and conferences!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## 📃 License

This repository is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)