<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-single-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-single.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-single.svg" width="70" height="70">
</picture>

# SwanLab: Open-Source Deep Learning Experiment Tracking and Visualization

**SwanLab empowers AI researchers with intuitive experiment tracking, real-time visualization, and collaborative features, making it easy to understand and optimize your machine learning models.** Explore your training runs with ease and efficiency by visiting the [original repo](https://github.com/SwanHubX/SwanLab).

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


</div>

<br/>

## Key Features

*   **Experiment Tracking**:  Log metrics, hyperparameters, and more with a simple Python API.
*   **Real-time Visualization**:  Interactive charts and dashboards to visualize training progress.
*   **Framework Integrations**: Seamlessly integrates with 30+ popular ML frameworks.
*   **Hardware Monitoring**: Track CPU, GPU, memory, and disk usage.
*   **Collaborative Features**: Share experiments, compare results, and collaborate with your team.
*   **Self-Hosted Option**: Run SwanLab locally or on your own infrastructure.
*   **Extensible with Plugins**: Customize SwanLab's functionality with plugins.

<br/>

## Table of Contents

-   [🌟 Recent Updates](#-最近更新)
-   [👋🏻 What is SwanLab](#-什么是swanlab)
-   [📃 Online Demo](#-在线演示)
-   [🏁 Quick Start](#-快速开始)
-   [💻 Self-Hosting](#-自托管)
-   [🔥 Practical Cases](#-实战案例)
-   [🎮 Hardware Monitoring](#-硬件记录)
-   [🚗 Framework Integrations](#-框架集成)
-   [🔌 Plugins and API](#-插件与api)
-   [🆚 Comparison with Familiar Tools](#-与熟悉的工具的比较)
-   [👥 Community](#-社区)
-   [📃 License](#-协议)

<br/>

## 🌟 Recent Updates

*   **(July 10, 2025):** 📚 More powerful **text view**, supporting Markdown rendering and arrow key switching, can be created by `swanlab.echarts.table` and `swanlab.Text`, [Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)

*   **(July 6, 2025):** 🚄 Support **resume breakpoint training**; New plugin **file recorder**; Integrated [ray](https://github.com/ray-project/ray) framework, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); Integrated [ROLL](https://github.com/volcengine/ROLL) framework, thanks to [@PanAndy](https://github.com/PanAndy), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

*   **(June 27, 2025):** 📊 Support **local amplification of small line charts**; Support configuration of **single line chart smoothing**; Significantly improved the interactive effect after the image chart is zoomed in;

*   **(June 20, 2025):** 🤗 Integrated [accelerate](https://github.com/huggingface/accelerate) framework, [PR](https://github.com/huggingface/accelerate/pull/3605), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html), enhanced the experimental record experience in distributed training;

*   **(June 18, 2025):** 🐜 Integrated [AREAL](https://github.com/inclusionAI/AReaL) framework, thanks to [@xichengpro](https://github.com/xichengpro), [PR](https://github.com/inclusionAI/AReaL/pull/98), [documentation](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process); 🖱 Support the mouse Hover to the sidebar experiment, highlighting the corresponding curve; Support cross-group comparison of line charts; Support setting experimental name cropping rules;

*   **(June 11, 2025):** 📊 Support **swanlab.echarts.table** data type, support pure text chart display; Support for **stretching interaction** of grouping to increase the number of charts displayed simultaneously; Table view adds **index maximum/minimum** options;

*   **(June 08, 2025):** ♻️ Support the local storage of complete experimental log files, upload local log files to the cloud/private deployment end through **swanlab sync**; Hardware monitoring supports **Hygon DCU**;

*   **(June 01, 2025):** 🏸 Support **free dragging of charts**; Support **ECharts custom charts**, adding more than 20 chart types including bar charts, pie charts, histograms; Hardware monitoring supports **MetaX GPU**; Integrated **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** framework;

*   **(May 25, 2025):** Logs support recording **standard error streams**, printing information from frameworks such as PyTorch Lightning can be better recorded; Hardware monitoring supports **Moore Threads**; New run command record security protection function, API Key will be automatically hidden;

<details><summary>Full Changelog</summary>

-   **(May 14, 2025):** Support **experimental Tag**; Support line chart **Log Scale**; Support **group dragging**; Greatly optimize the experience of uploading a large number of indicators; Add `swanlab.OpenApi` open interface;

-   **(May 09, 2025):** Support **line chart creation**; Configuration chart function adds **data source selection** function, supports single chart display different indicators; Support generation of **training project GitHub badge**;

-   **(April 23, 2025):** Support line chart **editing**, support free configuration of chart X and Y axis data range and title style; Chart search supports **regular expressions**; Support hardware detection and monitoring of **Kunlunxin XPU**;

-   **(April 11, 2025):** Support line chart **local area selection**; Support global selection of the step range of the dashboard line chart; Support one-click hide all charts;

-   **(April 08, 2025):** Support **swanlab.Molecule** data type, support recording and visualizing biochemical molecule data; Support saving the sort, filter, and column order change status in the table view;

-   **(April 07, 2025):** We have completed joint integration with [EvalScope](https://github.com/ModelScope/EvalScope), and now you can use SwanLab in EvalScope to **evaluate large model performance**;

-   **(March 30, 2025):** Support **swanlab.Settings** method, support more refined experimental behavior control; Support **Cambricon MLU** hardware monitoring; Support [Slack notification](https://docs.swanlab.cn/plugin/notification-slack.html), [Discord notification](https://docs.swanlab.cn/plugin/notification-discord.html);

-   **(March 21, 2025):** 🎉🤗 HuggingFace Transformers has officially integrated SwanLab (>=4.50.0 version), [#36433](https://github.com/huggingface/transformers/pull/36433); New **Object3D chart**, support recording and visualizing three-dimensional point cloud, [documentation](https://docs.swanlab.cn/api/py-object3d.html); Hardware monitoring supports GPU memory (MB), disk utilization, network uplink and downlink recording;

-   **(March 12, 2025):** 🎉🎉 SwanLab **private deployment version** is now available!! [🔗Deployment Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab now supports plugin extensions, such as [email notification](https://docs.swanlab.cn/plugin/notification-email.html), [Feishu notification](https://docs.swanlab.cn/plugin/notification-lark.html)

-   **(March 09, 2025):** Support **experimental sidebar widening**; Added a Git code button; Added **sync_mlflow** function, support for synchronizing experimental tracking with mlflow framework;

-   **(March 06, 2025):** We have completed joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio), and now you can use SwanLab in DiffSynth Studio to **track and visualize Diffusion model text-to-image/video experiments**, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html);

-   **(March 04, 2025):** New **MLFlow conversion** function, supports converting MLFlow experiments to SwanLab experiments, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html);

-   **(March 01, 2025):** Added **move experiment** function, you can now move experiments to different projects of different organizations;

-   **(February 24, 2025):** We have completed joint integration with [EasyR1](https://github.com/hiyouga/EasyR1), and now you can use SwanLab in EasyR1 to **track and visualize multimodal large model reinforcement learning experiments**, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)

-   **(February 18, 2025):** We have completed joint integration with [Swift](https://github.com/modelscope/ms-swift), and now you can use SwanLab in Swift's CLI/WebUI to **track and visualize large model fine-tuning experiments**, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html).

-   **(February 16, 2025):** Added **chart move grouping, create grouping** function.

-   **(February 09, 2025):** We have completed joint integration with [veRL](https://github.com/volcengine/verl), and now you can use SwanLab in veRL to **track and visualize large model reinforcement learning experiments**, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html).

-   **(February 05, 2025):** `swanlab.log` supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), adapting to Jax framework features; Supports `name` and `notes` parameters;

-   **(January 22, 2025):** Added `sync_tensorboardX` and `sync_tensorboard_torch` functions to support synchronizing experimental tracking with these two TensorBoard frameworks;

-   **(January 17, 2025):** Added `sync_wandb` function, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html), support for synchronizing experiment tracking with Weights & Biases; Greatly improved log rendering performance

-   **(January 11, 2025):** The cloud version greatly optimizes the performance of the project table and supports drag, sort, filter and other interactions

-   **(January 01, 2025):** Added line chart **persistent smoothing**, line chart drag-and-drop size change, optimized chart browsing experience

-   **(December 22, 2024):** We have completed joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), and now you can use SwanLab in LLaMA Factory to **track and visualize large model fine-tuning experiments**, [Usage Guide](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger).

-   **(December 15, 2024):** **Hardware monitoring (0.4.0)** function is online, supporting system-level information recording and monitoring of CPU, NPU (Ascend), GPU (Nvidia).

-   **(December 06, 2024):** Added integration with [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html) and [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html); Increased the limit on the length of a single line of log records.

</details>

<br>

## 👋🏻 What is SwanLab

SwanLab is an open-source, lightweight AI model training tracking and visualization tool designed for tracking, recording, comparing, and collaborating on experiments.

SwanLab provides an intuitive Python API and a beautiful UI interface for AI researchers, offering features such as **training visualization, automated logging, hyperparameter recording, experiment comparison, and multi-person collaboration**. With SwanLab, researchers can identify training issues based on intuitive visual charts, compare multiple experiments to find research inspiration, and break down the barriers to team communication through **online web page** sharing and **multi-person collaborative training** based on organizations, thereby improving organizational training efficiency.

https://github.com/user-attachments/assets/7965fec4-c8b0-4956-803d-dbf177b44f54

Here's a list of its core features:

**1. 📊 Experiment Metrics and Hyperparameter Tracking**: Embed simple code into your machine learning pipeline to track and record key training metrics.

-   ☁️ Supports **cloud** usage (similar to Weights & Biases), view training progress anytime, anywhere. [How to view experiments on your phone](https://docs.swanlab.cn/guide_cloud/general/app.html)

-   📝 Supports **hyperparameter recording**, **metric summary**, **table analysis**

-   🌸 **Visualize the training process**: Visualizing experiment tracking data through the UI interface allows trainers to intuitively see the results of each step of the experiment, analyze the trend of indicators, and determine which changes have led to improvements in the model effect, thereby improving the model iteration efficiency as a whole.

-   **Supported metadata types**: Scalar metrics, images, audio, text, 3D point clouds, biochemical molecules, Echarts custom charts...

![swanlab-table](readme_files/molecule.gif)

-   **Supported chart types**: Line charts, media charts (images, audio, text), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, [custom charts](https://docs.swanlab.cn/guide_cloud/experiment_track/log-custom-chart.html)...

[![swanlab-echarts](readme_files/echarts.png)](https://swanlab.cn/@ZeyiLin/swanlab-echarts-demo/charts)

-   **LLM Generated Content Visualization Component**: Text content visualization chart created for large language model training scenarios, supports Markdown rendering

![text-chart](readme_files/text-chart.gif)

-   **Automatic background recording**: Log logging, hardware environment, Git repository, Python environment, Python library list, project runtime directory

-   **Breakpoint training record**: Support supplementing new metric data to the same experiment after training is complete/interrupted

**2. ⚡️ Comprehensive Framework Integrations**: PyTorch, 🤗HuggingFace Transformers, PyTorch Lightning, 🦙LLaMA Factory, MMDetection, Ultralytics, PaddleDetetion, LightGBM, XGBoost, Keras, Tensorboard, Weights&Biases, OpenAI, Swift, XTuner, Stable Baseline3, Hydra, and **30+** frameworks

![](readme_files/integrations.png)

**3. 💻 Hardware Monitoring**: Supports real-time recording and monitoring of CPU, NPU (**Ascend**), GPU (**Nvidia**), MLU (**Cambricon**), XLU (**Kunlunxin**), DCU (**Hygon**), MetaX GPU (**Muxi XPU**), Moore Threads GPU (**Moore Threads**), and memory system-level hardware metrics

**4. 📦 Experiment Management**: Quickly manage multiple projects and experiments through a centralized dashboard designed specifically for training scenarios, with an overall view to quickly browse the global view

**5. 🆚 Compare Results**: Compare the hyperparameters and results of different experiments through online tables and comparison charts to explore iteration inspiration

![](readme_files/swanlab-table.png)

**6. 👥 Online Collaboration**: You can collaborate with your team for collaborative training, support real-time synchronization of experiments under one project, and you can view the team's training records online, make comments and suggestions based on the results

**7. ✉️ Share Results**: Copy and send permanent URLs to share each experiment, easily send them to partners, or embed them in online notes

**8. 💻 Self-Hosting Support**: Supports offline environment usage, the self-hosted community version can also view the dashboard and manage experiments, [Usage Guide](#-自托管)

**9. 🔌 Plugin Extensions**: Support extending the usage scenarios of SwanLab through plugins, such as [Feishu notification](https://docs.swanlab.cn/plugin/notification-lark.html), [Slack notification](https://docs.swanlab.cn/plugin/notification-slack.html), [CSV recorder](https://docs.swanlab.cn/plugin/writer-csv.html), etc.

> \[!IMPORTANT]
>
> **Star the project**, you will receive all release notifications from GitHub without delay ⭐️

![star-us](readme_files/star-us.png)

<br>

## 📃 Online Demo

Check out the online demos of SwanLab:

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Tracks image classification tasks by training a simple ResNet50 model on a cat and dog dataset. | Uses Yolov8 on the COCO128 dataset for object detection tasks, tracking training hyperparameters and metrics. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Tracks instruction fine-tuning of the Qwen2 large language model to complete simple instruction following. | Trains a simple LSTM model on the Google stock price dataset to predict future stock prices. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiments from ResNet to ResNeXt on audio classification tasks | Based on the Qwen2-VL multi-modal large model, performs Lora fine-tuning on the COCO2014 dataset. |

| [EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Training multimodal LLM RL using the EasyR1 framework | Training GRPO based on the Qwen2.5-0.5B model on the GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

If you want to experience the latest features, you can install from source.

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

### 2. Login and Get API Key

1.  [Register an account](https://swanlab.cn) for free.
2.  Log in to your account and copy your API Key in User Settings > [API Key](https://swanlab.cn/settings).
3.  Open the terminal and enter:

```bash
swanlab login
```

When prompted, enter your API Key and press Enter to log in.

### 3. Integrate SwanLab into your Code

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

You're all set! Go to [SwanLab](https://swanlab.cn) to view your first SwanLab experiment.

<br>

## 💻 Self-Hosting

The self-hosted community version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy the Self-hosted Version Using Docker

For details, please refer to: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Quick installation in China:

```bash
./install.sh
```

Install by pulling images from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Direct Experiments to the Self-hosted Service

Log in to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, you can record the experiment to the self-hosted service.

<br>

## 🔥 Practical Cases

**Excellent tutorial open source projects using SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): Large language model principle and practice tutorial from scratch ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm): "Open Source Large Model Guide" is tailored for Chinese babies, based on a Linux environment for quick fine-tuning (full parameter/Lora), deploying domestic and foreign open source large models (LLM)/multi-modal large models (MLLM) tutorials ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, expansion and reproduction. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)

**Tutorial Articles:**

*   [MNIST Handwriting Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pre-training](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multi-modal Large Model Fine-tuning Practical](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)

<br>

## 🎮 Hardware Monitoring

SwanLab records the **hardware information** and **resource usage** during AI training. The following is the supported table:

| Hardware | Information Recording | Resource Monitoring | Script |
| --- | --- | --- | --- |
| Nvidia GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
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

If you would like to record other hardware, welcome to submit Issue and PR!

<br>

## 🚗 Framework Integrations

Use your favorite framework with SwanLab!  
Here's a list of the frameworks we've integrated. Welcome to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to provide feedback on the frameworks you want to integrate.

**Basic Frameworks**

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized / Fine-tuning Frameworks**

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

*   [Stable Baseline3