<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: Open-Source Deep Learning Training Tracker and Visualizer

**SwanLab is a powerful, open-source tool designed to streamline deep learning training by providing comprehensive tracking, visualization, and collaboration features.** Easily integrate SwanLab into your projects to monitor experiments, analyze results, and collaborate with your team.  [Check out the original repo](https://github.com/SwanHubX/SwanLab) for more information.

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

[中文 / English / 日本語 / Русский]

👋 加入我们的[微信群](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking:** Log and visualize key metrics, hyperparameters, and more.
*   **Comprehensive Framework Integration:** Seamlessly integrates with 30+ popular ML frameworks.
*   **Hardware Monitoring:** Monitor CPU, GPU, memory usage, and other hardware metrics in real-time.
*   **Experiment Comparison:** Compare and analyze results across different experiments.
*   **Collaboration:** Facilitate team collaboration and knowledge sharing.
*   **Rich Data Types:** Supports logging of scalars, images, audio, text, videos, 3D point clouds, and more.
*   **Customizable Visualization:** Offers a wide range of chart types including line plots, media views, custom charts, and more.
*   **Self-Hosting:** Supports local and self-hosted deployments for enhanced control and privacy.
*   **Plugin Extensibility:** Extend functionality with plugins for notifications, data export, and more.

### Recent Updates

*   **2025.07.29:** 🚀 Sidebar supports **experiment filtering and sorting**; 📊 Table view features a **column control panel** for hiding and showing columns; 🔐 **Multi-API Key** management上线, making your data more secure;  new charts: PR curve, ROC curve, confusion matrix;
*   **2025.07.17:** 📊 Improved **line chart configuration**, supporting flexible configuration of line styles, colors, thicknesses, grids, and legend positions; 📹 Support for **swanlab.Video** data type, supporting recording and visualization of GIF files; Global chart dashboard supports configuring Y-axis and maximum display experiment count;
*   **2025.07.10:** 📚 More powerful **text view**, supports Markdown rendering and arrow key switching, can be created by `swanlab.echarts.table` and `swanlab.Text`;
*   **2025.07.06:** 🚄 Support **resume breakpoint training**; New plugin **file recorder**; Integrated with [ray](https://github.com/ray-project/ray) framework, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); Integrated with [ROLL](https://github.com/volcengine/ROLL) framework, thanks to [@PanAndy](https://github.com/PanAndy), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html);
*   **2025.06.27:** 📊 Support **zoom in on small line charts**; Support for configuring **smoothing for single line charts**; Significantly improved the interaction effect after image charts are enlarged;
*   **2025.06.20:** 🤗 Integrated with [accelerate](https://github.com/huggingface/accelerate) framework, [PR](https://github.com/huggingface/accelerate/pull/3605), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html), enhancing the experiment recording experience in distributed training;
*   **2025.06.18:** 🐜 Integrated with [AREAL](https://github.com/inclusionAI/AReaL) framework, thanks to [@xichengpro](https://github.com/xichengpro), [PR](https://github.com/inclusionAI/AReaL/pull/98), [documentation](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process); 🖱 Support mouse hover on the sidebar experiment, highlight the corresponding curve; Support cross-group comparison line charts; Support setting experiment name cropping rules;
*   **2025.06.11:** 📊 Support **swanlab.echarts.table** data type, supports pure text chart display; Support **stretching interaction** for grouping to increase the number of charts displayed at the same time; The table view adds the **index maximum/minimum value** option;
*   **2025.06.08:** ♻️ Support local storage of complete experiment log files, upload local log files to the cloud/privately deployed end through **swanlab sync**; Hardware monitoring supports **Hygon DCU**.

<details><summary>完整更新日志</summary>

- 2025.06.01：🏸支持**图表自由拖拽**；支持**ECharts自定义图表**，增加包括柱状图、饼状图、直方图在内的20+图表类型；硬件监控支持**沐曦GPU**；集成 **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** 框架；

- 2025.05.25：日志支持记录**标准错误流**，PyTorch Lightning等框架的打印信息可以被更好地记录；硬件监控支持**摩尔线程**；新增运行命令记录安全防护功能，API Key将被自动隐藏；

- 2025.05.14：支持**实验Tag**；支持折线图**Log Scale**；支持**分组拖拽**；大幅度优化了大量指标上传的体验；增加`swanlab.OpenApi`开放接口；

- 2025.05.09：支持**折线图创建**；配置图表功能增加**数据源选择**功能，支持单张图表显示不同的指标；支持生成**训练项目GitHub徽章**；

- 2025.04.23：支持折线图**编辑**，支持自由配置图表的X、Y轴数据范围和标题样式；图表搜索支持**正则表达式**；支持**昆仑芯XPU**的硬件检测与监控；

- 2025.04.11：支持折线图**局部区域选取**；支持全局选择仪表盘折线图的step范围；支持一键隐藏全部图表；

- 2025.04.08：支持**swanlab.Molecule**数据类型，支持记录与可视化生物化学分子数据；支持保存表格视图中的排序、筛选、列顺序变化状态；

- 2025.04.07：我们与 [EvalScope](https://github.com/ModelScope/EvalScope) 完成了联合集成，现在你可以在EvalScope中使用SwanLab来**评估大模型性能**；

- 2025.03.30：支持**swanlab.Settings**方法，支持更精细化的实验行为控制；支持**寒武纪MLU**硬件监控；支持 [Slack通知](https://docs.swanlab.cn/plugin/notification-slack.html)、[Discord通知](https://docs.swanlab.cn/plugin/notification-discord.html)；

- 2025.03.21：🎉🤗HuggingFace Transformers已正式集成SwanLab（>=4.50.0版本），[#36433](https://github.com/huggingface/transformers/pull/36433)；新增 **Object3D图表** ，支持记录与可视化三维点云，[文档](https://docs.swanlab.cn/api/py-object3d.html)；硬件监控支持了 GPU显存（MB）、磁盘利用率、网络上下行 的记录；

- 2025.03.12：🎉🎉SwanLab**私有化部署版**现已发布！！[🔗部署文档](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)；SwanLab 已支持插件扩展，如 [邮件通知](https://docs.swanlab.cn/plugin/notification-email.html)、[飞书通知](https://docs.swanlab.cn/plugin/notification-lark.html)

- 2025.03.09：支持**实验侧边栏拉宽**；新增外显 Git代码 按钮；新增 **sync_mlflow** 功能，支持与mlflow框架同步实验跟踪；

- 2025.03.06：我们与 [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio) 完成了联合集成，现在你可以在DiffSynth Studio中使用SwanLab来**跟踪和可视化Diffusion模型文生图/视频实验**，[使用指引](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html)；

- 2025.03.04：新增 **MLFlow转换** 功能，支持将MLFlow实验转换为SwanLab实验，[使用指引](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html)；

- 2025.03.01：新增 **移动实验** 功能，现在可以将实验移动到不同组织的不同项目下了；

- 2025.02.24：我们与 [EasyR1](https://github.com/hiyouga/EasyR1) 完成了联合集成，现在你可以在EasyR1中使用SwanLab来**跟踪和可视化多模态大模型强化学习实验**，[使用指引](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)

- 2025.02.18：我们与 [Swift](https://github.com/modelscope/ms-swift) 完成了联合集成，现在你可以在Swift的CLI/WebUI中使用SwanLab来**跟踪和可视化大模型微调实验**，[使用指引](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html)。

- 2025.02.16：新增 **图表移动分组、创建分组** 功能。

- 2025.02.09：我们与 [veRL](https://github.com/volcengine/verl) 完成了联合集成，现在你可以在veRL中使用SwanLab来**跟踪和可视化大模型强化学习实验**，[使用指引](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)。

- 2025.02.05：`swanlab.log`支持嵌套字典 [#812](https://github.com/SwanHubX/SwanLab/pull/812)，适配Jax框架特性；支持`name`与`notes`参数；

- 2025.01.22：新增`sync_tensorboardX`与`sync_tensorboard_torch`功能，支持与此两种TensorBoard框架同步实验跟踪；

- 2025.01.17：新增`sync_wandb`功能，[文档](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html)，支持与Weights & Biases实验跟踪同步；大幅改进了日志渲染性能

- 2025.01.11：云端版大幅优化了项目表格的性能，并支持拖拽、排序、筛选等交互

- 2025.01.01：新增折线图**持久化平滑**、折线图拖拽式改变大小，优化图表浏览体验

- 2024.12.22：我们与 [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 完成了联合集成，现在你可以在LLaMA Factory中使用SwanLab来**跟踪和可视化大模型微调实验**，[使用指引](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger)。

- 2024.12.15：**硬件监控（0.4.0）** 功能上线，支持CPU、NPU（Ascend）、GPU（Nvidia）的系统级信息记录与监控。

- 2024.12.06：新增对[LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html)、[XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html)的集成；提高了对日志记录单行长度的限制。

- 2024.11.26：环境选项卡-硬件部分支持识别**华为昇腾NPU**与**鲲鹏CPU**；云厂商部分支持识别青云**基石智算**。

</details>

<br>

## Online Demos

Explore the power of SwanLab with these interactive demos:

| [ResNet50 猫狗分类][demo-cats-dogs] | [Yolov8-COCO128 目标检测][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Track a simple ResNet50 model on a cat and dog dataset for image classification tasks. | Use Yolov8 to track hyper-parameters and metrics for object detection on the COCO128 dataset. |

| [Qwen2 指令微调][demo-qwen2-sft] | [LSTM Google 股票预测][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Track instruction fine-tuning training on the Qwen2 large language model to complete simple instruction following. | Use a simple LSTM model to train on the Google stock price dataset to predict future stock prices. |

| [ResNeXt101 音频分类][demo-audio-classification] | [Qwen2-VL COCO数据集微调][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiment process from ResNet to ResNeXt on audio classification tasks | Fine-tuning Lora based on Qwen2-VL multi-modal large model on COCO2014 dataset |

| [EasyR1 多模态LLM RL训练][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO训练][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Using EasyR1 framework for multi-modal LLM RL training | Train GRPO on the GSM8k dataset based on the Qwen2.5-0.5B model |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## Getting Started

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

```bash
# Option 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Option 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Installation</summary>

[Offline Dashboard Documentation](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get Your API Key

1.  [Register an account](https://swanlab.cn) for free.
2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and enter:

```bash
swanlab login
```

When prompted, enter your API Key and press Enter to complete the login.

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

Congratulations!  View your first SwanLab experiment at [SwanLab](https://swanlab.cn).

<br>

## Self-Hosting

The self-hosted community version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy the Self-Hosted Version with Docker

Refer to the following documents for details: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

China region quick installation:

```bash
./install.sh
```

Pull the image from DockerHub to install:

```bash
./install-dockerhub.sh
```

### 2. Assign the Experiment to the Self-Hosted Service

Log in to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, you can record the experiment to the self-hosted service.

<br>

## Practical Case Studies

**Excellent tutorial open source project using SwanLab:**
- [happy-llm](https://github.com/datawhalechina/happy-llm): Large language model principle and practice tutorial from scratch ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
- [self-llm](https://github.com/datawhalechina/self-llm): "Open Source Large Model Eating Guide" tailored for Chinese babies, Linux environment-based fast fine-tuning (full parameter/Lora), deploying domestic and foreign open source large models (LLM)/multi-modal large models (MLLM) tutorial ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
- [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, expansion and replication.![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)

**Excellent papers using SwanLab:**
- [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
- [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)

**Tutorial Articles:**
- [MNIST Handwriting Recognition](https://docs.swanlab.cn/examples/mnist.html)
- [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
- [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
- [Resnet Cat and Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
- [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
- [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
- [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
- [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
- [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
- [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
- [Stable Diffusion Text-to-Image Fine-Tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
- [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
- [GLM4 Instruction Fine-Tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
- [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
- [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
- [Qwen3 Medical Model Fine-Tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
- [Qwen2-VL Multi-modal Large Model Fine-tuning Practice](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
- [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
- [Qwen3-SmVL-0.6B Multi-modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
- [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
- [GLM-4.5-Air-LoRA and SwanLab Visualization Record](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)

🌟 If you have any tutorials you want to include, welcome to submit a PR!

<br>

## Hardware Recording

SwanLab records **hardware information** and **resource usage** during AI training, here is the support status table:

| Hardware | Information Recording | Resource Monitoring | Script |
| --- | --- | --- | --- |
| NVIDIA GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
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

If you want to record other hardware, please submit an Issue and PR!

<br>

## Framework Integrations

Use your favorite framework with SwanLab!  
Here is a list of frameworks we have integrated, and we welcome you to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to give feedback on the framework you want to integrate.

**Basic Frameworks**
- [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
- [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
- [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Proprietary/Fine-tuning Frameworks**
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

<br>

## Plugins and API

Extend SwanLab's functionality with plugins to enhance your experiment management experience!

-   [Customize Your Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email Notification](https://docs.swanlab.cn/plugin/notification-email.html)
-   [Feishu Notification](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notification](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [Enterprise WeChat Notification](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord Notification](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack Notification](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV Recorder](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [File Recorder](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open APIs:
-   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparison with Familiar Tools

### TensorBoard vs. SwanLab

*   **☁️ Online Usage:** SwanLab provides convenient online synchronization and saving of training experiments in the cloud, making it easy to remotely view training progress, manage historical projects, share experiment links, send real-time message notifications, view experiments on multiple devices, and more. Tensorboard is an offline