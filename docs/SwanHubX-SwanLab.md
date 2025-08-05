<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## SwanLab: The Open-Source Tool for Deep Learning Experiment Tracking and Visualization

SwanLab is an open-source platform that simplifies deep learning experiment tracking, offering a modern UI and seamless integration with 30+ popular frameworks for cloud and offline use. [Check out the original repo on GitHub!](https://github.com/SwanHubX/SwanLab)

<a href="https://swanlab.cn">🔥SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Documentation</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks</a>

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

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Table of Contents

- [🌟 Recent Updates](#-recent-updates)
- [👋🏻 What is SwanLab?](#-what-is-swanlab)
- [📃 Online Demo](#-online-demo)
- [🏁 Quickstart](#-quickstart)
- [💻 Self-Hosting](#-self-hosting)
- [🔥 Real-World Examples](#-real-world-examples)
- [🎮 Hardware Monitoring](#-hardware-monitoring)
- [🚗 Framework Integrations](#-framework-integrations)
- [🔌 Plugins and API](#-plugins-and-api)
- [🆚 Comparison with Similar Tools](#-comparison-with-similar-tools)
- [👥 Community](#-community)
- [📃 License](#-license)

<br/>

## 🌟 Recent Updates

- **2025.07.29:** 🚀 Sidebar now supports **experiment filtering and sorting**; 📊 Table view introduces a **column control panel** for easy column hiding/showing; 🔐 **Multi-API Key** management is live for enhanced data security; swanlab sync improves compatibility for log file integrity, adapting to scenarios like training crashes; New charts - PR curve, ROC curve, and confusion matrix are launched, [Docs](https://docs.swanlab.cn/api/py-pr_curve.html);

- **2025.07.17:** 📊 More powerful **line chart configuration**, supporting flexible configuration of line types, colors, thicknesses, grids, and legend positions; 📹 Supports **swanlab.Video** data type, supports recording and visualizing GIF format files; Global chart dashboard supports configuring the Y-axis and the maximum number of experiments displayed;

- **2025.07.10:** 📚 More powerful **text view**, supports Markdown rendering and arrow key switching, can be created by `swanlab.echarts.table` and `swanlab.Text`, [Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)

- **2025.07.06:** 🚄 Supports **resume breakpoint training**; New plugin **file recorder**; Integrated [ray](https://github.com/ray-project/ray) framework, [Docs](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); Integrated [ROLL](https://github.com/volcengine/ROLL) framework, thanks to [@PanAndy](https://github.com/PanAndy), [Docs](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

- **2025.06.27:** 📊 Supports **local zoom for small line charts**; Supports configuring **smoothing for a single line chart**; Significantly improved the interactive effects after image charts are zoomed in;

- **2025.06.20:** 🤗 Integrated [accelerate](https://github.com/huggingface/accelerate) framework, [PR](https://github.com/huggingface/accelerate/pull/3605), [Docs](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html), enhancing the experimental logging experience in distributed training;

- **2025.06.18:** 🐜 Integrated [AREAL](https://github.com/inclusionAI/AReaL) framework, thanks to [@xichengpro](https://github.com/xichengpro), [PR](https://github.com/inclusionAI/AReaL/pull/98), [Docs](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process); 🖱 Supports highlighting corresponding curves when hovering the mouse over experiments in the sidebar; Supports cross-group comparison of line charts; Supports setting experiment name cropping rules;

- **2025.06.11:** 📊 Support **swanlab.echarts.table** data type, support plain text chart display; Support for **stretching interaction** of groups to increase the number of charts displayed at the same time; The table view adds a **metric maximum/minimum** option;

- **2025.06.08:** ♻️ Supports storing complete experiment log files locally and uploading local log files to the cloud/private deployment end through **swanlab sync**; Hardware monitoring supports **Hygon DCU**;

<details><summary>Complete Changelog</summary>

- 2025.06.01：🏸Support **free drag and drop of charts**; Support **ECharts custom charts**, adding 20+ chart types including bar charts, pie charts, and histograms; Hardware monitoring supports **Muxi GPU**; Integrated **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** framework;

- 2025.05.25：Logs support recording **standard error streams**, and print information from frameworks like PyTorch Lightning can be better recorded; Hardware monitoring supports **Moore Threads**; Added a security protection function for recording run commands, and the API Key will be automatically hidden;

- 2025.05.14：Support **experiment Tag**; Support line chart **Log Scale**; Support **group dragging**; Significantly optimized the experience of uploading a large number of metrics; Added `swanlab.OpenApi` open interface;

- 2025.05.09：Support **line chart creation**; The chart configuration function adds a **data source selection** function, supporting different indicators to be displayed in a single chart; Support generating **training project GitHub badges**;

- 2025.04.23：Support line chart **editing**, support free configuration of the X and Y axis data range and title style; Chart search supports **regular expressions**; Support hardware detection and monitoring of **Kunlunxin XPU**;

- 2025.04.11：Support line chart **local area selection**; Support global selection of the step range of the dashboard line chart; Support one-click hiding of all charts;

- 2025.04.08：Support **swanlab.Molecule** data type, support recording and visualizing biochemical molecular data; Support saving the sorting, filtering, and column order changes in the table view;

- 2025.04.07：We have completed a joint integration with [EvalScope](https://github.com/ModelScope/EvalScope), and now you can use SwanLab in EvalScope to **evaluate large model performance**;

- 2025.03.30：Support **swanlab.Settings** method, support more refined experiment behavior control; Support **Cambricon MLU** hardware monitoring; Support [Slack notification](https://docs.swanlab.cn/plugin/notification-slack.html), [Discord notification](https://docs.swanlab.cn/plugin/notification-discord.html);

- 2025.03.21：🎉🤗HuggingFace Transformers has officially integrated SwanLab (>=4.50.0 version), [#36433](https://github.com/huggingface/transformers/pull/36433); Added **Object3D chart**, support recording and visualizing 3D point cloud, [Docs](https://docs.swanlab.cn/api/py-object3d.html); Hardware monitoring supports the recording of GPU video memory (MB), disk utilization, and network upstream and downstream;

- 2025.03.12：🎉🎉SwanLab **private deployment version** has been released!! [🔗Deployment document](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab has supported plugin extension, such as [Email notification](https://docs.swanlab.cn/plugin/notification-email.html), [Feishu notification](https://docs.swanlab.cn/plugin/notification-lark.html)

- 2025.03.09：Support **experiment sidebar widening**; Added a button for external display of Git code; Added **sync_mlflow** function, support for synchronizing experiment tracking with the mlflow framework;

- 2025.03.06：We have completed a joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio), and now you can use SwanLab in DiffSynth Studio to **track and visualize Diffusion model text-to-image/video experiments**, [Instructions for use](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html);

- 2025.03.04：Added **MLFlow conversion** function, support converting MLFlow experiments to SwanLab experiments, [Instructions for use](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html);

- 2025.03.01：Added **move experiment** function, now you can move experiments to different projects of different organizations;

- 2025.02.24：We have completed a joint integration with [EasyR1](https://github.com/hiyouga/EasyR1), and now you can use SwanLab in EasyR1 to **track and visualize multimodal large model reinforcement learning experiments**, [Instructions for use](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)

- 2025.02.18：We have completed a joint integration with [Swift](https://github.com/modelscope/ms-swift), and now you can use SwanLab in the CLI/WebUI of Swift to **track and visualize large model fine-tuning experiments**, [Instructions for use](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html).

- 2025.02.16：Added **chart move grouping, create grouping** function.

- 2025.02.09：We have completed a joint integration with [veRL](https://github.com/volcengine/verl), and now you can use SwanLab in veRL to **track and visualize large model reinforcement learning experiments**, [Instructions for use](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html).

- 2025.02.05：`swanlab.log` supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), adapting to Jax framework characteristics; supports `name` and `notes` parameters;

- 2025.01.22：Added `sync_tensorboardX` and `sync_tensorboard_torch` functions, supports synchronizing experiment tracking with these two TensorBoard frameworks;

- 2025.01.17：Added `sync_wandb` function, [Docs](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html), supports synchronization of experiment tracking with Weights & Biases; Significantly improved log rendering performance

- 2025.01.11：The cloud version significantly optimized the performance of project tables, and supports interactions such as drag and drop, sorting, and filtering

- 2025.01.01：Added line chart **persistent smoothing**, line chart drag-and-drop to change size, and optimized the chart browsing experience

- 2024.12.22：We have completed a joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), and now you can use SwanLab in LLaMA Factory to **track and visualize large model fine-tuning experiments**, [Instructions for use](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger).

- 2024.12.15：**Hardware monitoring (0.4.0)** function launched, supporting system-level information recording and monitoring of CPU, NPU (Ascend), and GPU (Nvidia).

- 2024.12.06：Added integration with [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html), [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html); Increased the limit on the length of a single line of log recording.

- 2024.11.26：The environment tab - hardware section supports identifying **Huawei Ascend NPU** and **Kunpeng CPU**; The cloud vendor section supports identifying Qingyun **BaseStone Intelligent Computing**.

</details>

<br>

## 👋🏻 What is SwanLab?

SwanLab is an open-source, lightweight tool designed for tracking and visualizing AI model training, offering a comprehensive platform for tracking, recording, comparing, and collaborating on experiments.

SwanLab caters to AI researchers by providing a user-friendly Python API and an elegant UI, offering features such as **training visualization, automated logging, hyperparameter recording, experiment comparison, and multi-user collaboration**. Researchers can identify training issues based on intuitive visualizations, compare multiple experiments to spark new ideas, and break down team communication barriers through **online web pages** and **multi-user training** based on organizations, thereby improving organizational training efficiency.

https://github.com/user-attachments/assets/7965fec4-c8b0-4956-803d-dbf177b44f54

Here's a list of its core features:

**1. 📊 Experiment Metrics and Hyperparameter Tracking**: Integrate simple code into your machine learning pipeline to track and record key training metrics.

- ☁️ **Cloud-Based Usage**: View training progress from anywhere, similar to Weights & Biases. [How to view experiments on your phone](https://docs.swanlab.cn/guide_cloud/general/app.html)

- 📝 **Hyperparameter Logging, Metric Summarization, and Table Analysis**

- 🌸 **Training Process Visualization**: The UI visualizes experimental tracking data, enabling trainers to intuitively see the results of each step of the experiment, analyze metric trends, and determine which changes have led to improvements in model performance, thereby improving the overall efficiency of model iteration.

- **Supported Metadata Types**: Scalar metrics, images, audio, text, videos, 3D point clouds, biochemical molecules, Echarts custom charts...

![swanlab-table](readme_files/molecule.gif)

- **Supported Chart Types**: Line charts, media charts (images, audio, text, videos), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, [Custom Charts](https://docs.swanlab.cn/guide_cloud/experiment_track/log-custom-chart.html)...

[![swanlab-echarts](readme_files/echarts.png)](https://swanlab.cn/@ZeyiLin/swanlab-echarts-demo/charts)

- **LLM Content Visualization Component**: A text content visualization chart designed for large language model training scenarios, supporting Markdown rendering

![text-chart](readme_files/text-chart.gif)

- **Background Auto-Logging**: Log logging, hardware environment, Git repository, Python environment, Python library list, project run directory

- **Breakpoint Resume Recording**: Supports supplementing new metric data to the same experiment after training completion/interruption

**2. ⚡️ Comprehensive Framework Integration**: Supports over **30+** frameworks, including PyTorch, 🤗HuggingFace Transformers, PyTorch Lightning, 🦙LLaMA Factory, MMDetection, Ultralytics, PaddleDetetion, LightGBM, XGBoost, Keras, Tensorboard, Weights&Biases, OpenAI, Swift, XTuner, Stable Baseline3, and Hydra.

![](readme_files/integrations.png)

**3. 💻 Hardware Monitoring**: Real-time recording and monitoring of system-level hardware metrics for CPU, NPU (**Ascend**), GPU (**Nvidia**), MLU (**Cambricon**), XLU (**Kunlunxin**), DCU (**Hygon**), MetaX GPU (**Muxi XPU**), Moore Threads GPU (**Moore Threads**), and memory.

**4. 📦 Experiment Management**: Manage multiple projects and experiments quickly with a centralized dashboard designed specifically for training scenarios, providing an overall view.

**5. 🆚 Result Comparison**: Compare hyperparameters and results of different experiments through online tables and comparison charts to discover iteration inspirations

![](readme_files/swanlab-table.png)

**6. 👥 Online Collaboration**: You can collaborate on training with your team, supporting real-time synchronization of experiments under a project, allowing you to view your team's training records online, and provide opinions and suggestions based on the results.

**7. ✉️ Share Results**: Copy and send persistent URLs to share each experiment, making it easy to send them to partners or embed them in online notes

**8. 💻 Self-Hosting Support**: Supports offline environments. The self-hosted community version can also view the dashboard and manage experiments. [Self-Hosting Guide](#-self-hosting)

**9. 🔌 Plugin Extension**: Support extending the usage scenarios of SwanLab through plugins, such as [Feishu notification](https://docs.swanlab.cn/plugin/notification-lark.html), [Slack notification](https://docs.swanlab.cn/plugin/notification-slack.html), [CSV recorder](https://docs.swanlab.cn/plugin/writer-csv.html), etc.

> \[!IMPORTANT]
>
> **Star the project** to receive all release notifications from GitHub without delay! ⭐️

![star-us](readme_files/star-us.png)

<br>

## 📃 Online Demo

Explore SwanLab's functionality with these online demos:

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Tracks a simple ResNet50 model trained on a cat/dog dataset for image classification. | Tracks training hyperparameters and metrics using Yolov8 on the COCO128 dataset for object detection. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Tracks instruction fine-tuning of the Qwen2 large language model to complete simple instruction following. | Uses a simple LSTM model to train on Google's stock price dataset to predict future stock prices. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiment from ResNet to ResNeXt on audio classification tasks | Lora fine-tuning based on Qwen2-VL multi-modal large model on COCO2014 dataset. |

| [EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Use EasyR1 framework for multi-modal LLM RL training | Based on the Qwen2.5-0.5B model, GRPO training on the GSM8k dataset |

[More examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

To experience the latest features, install from the source.

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

### 2. Login and Get Your API Key

1.  [Register for a free account](https://swanlab.cn)

2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).

3.  Open your terminal and enter:

```bash
swanlab login
```

Enter your API Key when prompted, then press Enter to complete the login.

### 3. Integrate SwanLab with Your Code

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

Congratulations! Head to [SwanLab](https://swanlab.cn) to view your first SwanLab experiment.

<br>

## 💻 Self-Hosting

The self-hosted community version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy the self-hosted version using Docker

For more details, please refer to: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Quick installation in China:

```bash
./install.sh
```

Installation by pulling images from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point experiments to the self-hosted service

Log in to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, you can record experiments to the self-hosted service.

<br>

## 🔥 Real-World Examples

**Excellent Tutorial Open Source Projects Using SwanLab:**
- [happy-llm](https://github.com/datawhalechina/happy-llm): Principles and practical tutorials of large language models from scratch ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
- [self-llm](https://github.com/datawhalechina/self-llm): "Open Source Large Model User Guide" tailored for Chinese users, a tutorial for quick fine-tuning (full parameter/Lora) and deploying domestic and international open source large models (LLM)/multimodal large models (MLLM) on Linux environments. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
- [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): Interpretation, extension, and reproduction of the DeepSeek series. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)

**Excellent Papers Using SwanLab:**
- [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
- [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)

**Tutorial Articles:**
- [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
- [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
- [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
- [Resnet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
- [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
- [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
- [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
- [DQN Reinforcement Learning - Cartpole Inverted Pendulum](https://docs.swanlab.cn/examples/dqn_cartpole.html)
- [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
- [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
- [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
- [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
- [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
- [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
- [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
- [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
- [Qwen2-VL Multi-modal Large Model Fine-tuning Practical](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
- [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
- [Qwen3-SmVL-0.6B Multi-modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
- [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
- [GLM-4.5-Air-LoRA and SwanLab Visualization Recording](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)

🌟If you have tutorials you'd like to include, welcome to submit a PR!

<br>

## 🎮 Hardware Monitoring

SwanLab records **hardware information** and **resource usage** during AI training. The following table shows the support status:

| Hardware | Information Recording | Resource Monitoring | Script |
|---|---|---|---|
| Nvidia GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU | ✅ | ✅ | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC | ✅ | ✅ | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU | ✅ | ✅ | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅ | ✅ | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Muxi GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU | ✅ | ✅ | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory | ✅ | ✅ | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk | ✅ | ✅ | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

If you want to record other hardware, welcome to submit an Issue and PR!

<br>

## 🚗 Framework Integrations

Use SwanLab with your favorite frameworks!  
Here's a list of the frameworks we have integrated, and we welcome you to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to provide feedback on the frameworks you'd like to see integrated.

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