<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: Open-Source Deep Learning Experiment Tracking & Visualization

**SwanLab is an open-source, user-friendly tool designed to streamline your deep learning workflow by tracking, visualizing, and collaborating on your machine learning experiments.**  Seamlessly integrate with 30+ popular frameworks and enjoy features like cloud/offline support, robust visualization, and powerful experiment comparison.  [Explore the project on GitHub](https://github.com/SwanHubX/SwanLab).

<a href="https://swanlab.cn">🔥SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Documentation</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks</a>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)


<br/>

![](readme_files/swanlab-overview.png)

[中文 / English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking & Visualization:**
    *   Track and visualize key metrics, hyperparameters, and model performance with intuitive UI.
    *   Support for various data types: Scalar metrics, images, audio, text, videos, 3D point clouds, biochemical molecules, and custom ECharts charts.
    *   Built-in support for common chart types: Line charts, media (images, audio, text, videos), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, and radar charts.
    *   Enhanced text visualization components tailored for LLM training scenarios, including Markdown rendering.
    *   Automatic logging of experiment metadata: Log files, hardware environment, Git repository, Python environment, Python library list, and project run directory.
    *   Resume training support - add new metrics to the same experiment, even after interruption.

*   **Comprehensive Framework Integrations:**  Seamless integration with over 30 popular machine learning frameworks, including PyTorch, Hugging Face Transformers, PyTorch Lightning, LLaMA Factory, and many more.

*   **Hardware Monitoring:**  Real-time monitoring and logging of system-level hardware metrics: CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (MXGPU), Moore Threads GPU, and memory usage.

*   **Experiment Management:**  Centralized dashboard designed for AI training, allowing for a global view to manage multiple projects and experiments.

*   **Result Comparison:**  Compare experiments using online tables and comparison charts to identify insights and optimize your model iterations.

*   **Online Collaboration:** Facilitate collaborative training with your team by syncing experiments in a project, where you can view training logs, discuss results, and provide feedback.

*   **Shareable Results:**  Share your experiment results effortlessly with shareable URLs.

*   **Self-Hosted Support:** Use SwanLab in offline environments through self-hosting, enabling you to manage experiments via the dashboard.

*   **Plugin Extensibility:** Extend SwanLab's functionality with plugins, such as those for notifications (Slack, Lark), and CSV recording.

>   \[!IMPORTANT]
>
>   **Star the repository** to receive the latest release notifications directly on GitHub. ⭐️

![star-us](readme_files/star-us.png)

<br/>

## Recent Updates

*   **2025.09.22:** 📊 New UI released, table view supports global sorting and filtering, data level unifies the table and chart view.
*   **2025.09.12:** 🔢 Support for creating **scalar charts**, flexibility in displaying statistical values for experimental metrics; Major upgrade to organization management page, providing more powerful permission control and project management capabilities.
*   **2025.08.19:** 🤔 Enhanced chart rendering performance and low-invasive loading animation, allowing researchers to focus more on experiment analysis; Integrated the excellent [MLX-LM](https://github.com/ml-explore/mlx-lm) and [SpecForge](https://github.com/sgl-project/SpecForge) frameworks, providing training experiences in more scenarios.
*   **2025.08.06:** 👥 **Training Collaboration** is online, supporting inviting project collaborators, sharing project links and QR codes; The workspace supports list view, supports displaying project Tags;
*   **2025.07.29:** 🚀 Sidebar supports **experiment filtering and sorting**; 📊 Table view is online with a **column control panel**, which can easily achieve column hiding and display; 🔐 **Multi-API Key** management is online, making your data more secure; swanlab sync improves the compatibility of log file integrity, adapts to scenarios such as training crashes; New charts - PR curve, ROC curve, confusion matrix online, [Documentation](https://docs.swanlab.cn/api/py-pr_curve.html);
*   **2025.07.17:** 📊 More powerful **line chart configuration**, supports flexible configuration of line types, colors, thickness, grids, and legend positions; 📹 Supports **swanlab.Video** data type, supports recording and visualizing GIF format files; Global chart dashboard supports configuring the Y-axis and maximum number of displayed experiments;
*   **2025.07.10:** 📚 More powerful **text view**, supports Markdown rendering and arrow key switching, which can be created by `swanlab.echarts.table` and `swanlab.Text`, [Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)
*   **2025.07.06:** 🚄 Support **resume breakpoint training**; New plugin **file recorder**; Integrated [ray](https://github.com/ray-project/ray) framework, [Documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); Integrated [ROLL](https://github.com/volcengine/ROLL) framework, thanks to [@PanAndy](https://github.com/PanAndy), [Documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)
*   **2025.06.27:** 📊 Support for **local zoom-in on small line charts**; Support for configuring **smoothing for a single line chart**; Significantly improved the interactive effect after the image chart is enlarged;

<details><summary>Complete Changelog</summary>

*   ... (rest of the changelog from original README)
</details>

<br/>

## Online Demo

Explore SwanLab in action with these interactive demos:

| [ResNet50 Cats vs. Dogs][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Tracks a ResNet50 model on a cats vs dogs image classification task. | Tracks a Yolov8 model on COCO128 dataset. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Tracks instruction fine-tuning of a Qwen2 LLM. | Predicts future stock prices using an LSTM model trained on Google stock data. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiments with ResNet to ResNeXt for audio classification tasks. | Fine-tunes the Qwen2-VL multimodal LLM on the COCO2014 dataset using LoRA. |

| [EasyR1 Multimodal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| RL training of a multimodal LLM with EasyR1 framework. | GRPO training of the Qwen2.5-0.5B model on the GSM8k dataset. |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br/>

## Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Extension</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get Your API Key

1.  [Register](https://swanlab.cn) for a free account.

2.  Log in and copy your API key from User Settings > [API Key](https://swanlab.cn/settings).

3.  Open your terminal and run:

    ```bash
    swanlab login
    ```

    Enter your API key when prompted and press Enter.

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

Now, head over to [SwanLab](https://swanlab.cn) to view your first experiment.

<br/>

## Self-Hosting

The self-hosted community version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploying a Self-Hosted Version Using Docker

Refer to the documentation for details: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For quick installation in China:

```bash
./install.sh
```

Installation by pulling the image from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Direct Experiments to Your Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, experiment logs will be recorded in your self-hosted service.

<br/>

## Practical Use Cases

**Open-source tutorials using SwanLab:**
- [happy-llm](https://github.com/datawhalechina/happy-llm) -  A tutorial on the principles and practice of large language models (LLMs) from scratch.
- [self-llm](https://github.com/datawhalechina/self-llm) - This "Guide to Open Source Large Models" provides a tutorial for fine-tuning (full parameter/Lora) and deploying open source large language models (LLMs) and multimodal large models (MLLMs) specifically tailored for Chinese users, based on a Linux environment.
- [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek) -  Analysis, extensions, and reproductions of the DeepSeek series.
- [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL) - Fine-tuned the visual head of SmolVLM2 with the Qwen3-0.6B model.
- [OPPO/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models) - End-to-end agent foundation models with multi-agent distillation and agent RL.

**Papers Utilizing SwanLab:**
- [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
- [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
- [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
- [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorial Articles:**
- [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
- [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
- [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
- [Resnet Cats and Dogs Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
- [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
- [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
- [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
- [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
- [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
- [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
- [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
- [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
- [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
- [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
- [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
- [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
- [Qwen2-VL Multimodal Model Fine-tuning Practice](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
- [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
- [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
- [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
- [GLM-4.5-Air-LoRA and SwanLab Visualization Record](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
- [How to do RAG? Open-sourced SwanLab document assistant solution](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟 Welcome to submit a PR if you want to include tutorials!

<br/>

## Hardware Monitoring

SwanLab records **hardware information** and **resource usage** during AI training.  Here's the hardware support table:

| Hardware | Information Recording | Resource Monitoring | Script |
| :--- | :--- | :--- | :--- |
| NVIDIA GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU | ✅ | ✅ | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC | ✅ | ✅ | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU | ✅ | ✅ | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅ | ✅ | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU | ✅ | ✅ | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory | ✅ | ✅ | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk | ✅ | ✅ | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Contribute by creating issues and PRs if you would like to add support for other hardware.

<br/>

## Framework Integrations

Integrate your favorite frameworks with SwanLab!
Here's a list of the integrated frameworks.  Feel free to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to request a new integration.

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

<br/>

## Plugin & API

Extend SwanLab with plugins!

- [Customize Your Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
- [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
- [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
- [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
- [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
- [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
- [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
- [CSV Recorder](https://docs.swanlab.cn/plugin/writer-csv.html)
- [File Recorder](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI:
- [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br/>

## Comparison with Similar Tools

### TensorBoard vs. SwanLab

-   **☁️ Online Support:** SwanLab offers cloud-based experiment tracking and convenient synchronization, enabling remote progress monitoring, project management, experiment sharing, real-time notifications, and multi-device access.  TensorBoard is designed primarily for offline experiment tracking.

-   **👥 Collaboration:** SwanLab streamlines collaborative machine learning efforts, especially for multi-team or cross-team projects, enabling easy management, experiment sharing, and discussion. TensorBoard, however, focuses on individual use cases.

-   **💻 Persistent, Centralized Dashboard:**  Regardless of where you train – local machine, lab cluster, or cloud GPU instance – your results are consistently recorded in a single, centralized dashboard. Managing and copying TFEvent files across machines takes time in TensorBoard.

-   **💪 Enhanced Tables:**  SwanLab's tables facilitate easy viewing, searching, and filtering of results from varied experiments, simplifying the task of exploring numerous model versions to identify top-performing models across different tasks. TensorBoard isn't optimized for large projects.

### Weights and Biases vs. SwanLab

-   Weights and Biases is a proprietary, cloud-based MLOps platform requiring an internet connection.

-   SwanLab offers both cloud and open-source, free, self-hosted options, as well as an entirely offline version.

<br/>

## Community

### Related Repositories

-   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting deployment scripts
-   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation
-   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard):  Lightweight offline dashboard web code opened by `swanlab watch`

### Community and Support

-   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report issues and ask questions.
-   [Email Support](zeyi.lin@swanhub.co): Contact with any feedback or inquiries about using SwanLab.
-   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Engage in discussions and ask questions.

### SwanLab README Badges

Add SwanLab badges to your project's README:

[![Tracking SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in your Paper

If SwanLab has been useful for your research, please cite it using this format:

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

Interested in contributing to SwanLab?  Please read the [contribution guide](CONTRIBUTING.md) first.

We also greatly appreciate support through social media, events, and conferences!

<br/>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br/>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## License

SwanLab is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE