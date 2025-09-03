<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br/>

## SwanLab: Effortlessly Track and Visualize Your Deep Learning Experiments

SwanLab is an open-source, modern deep learning experiment tracking and visualization tool that empowers researchers and engineers to understand, compare, and collaborate on their machine learning projects.  [Explore the SwanLab GitHub Repo](https://github.com/SwanHubX/SwanLab).

**Key Features:**

*   **📊 Rich Visualization**: Visualize training metrics, model performance, and hardware usage with interactive charts and dashboards.
*   **📝 Automated Logging**: Automatically track and record hyperparameters, metrics, and system information without extensive code changes.
*   **💻 Cloud & Offline Support**: Use SwanLab in the cloud or locally for maximum flexibility, with a user-friendly interface.
*   **🤝 Collaborative Features**: Facilitate team collaboration with project sharing and real-time experiment tracking.
*   **🔄 Broad Framework Compatibility**: Seamlessly integrate with over 30 popular deep learning frameworks, including PyTorch, TensorFlow, and Hugging Face.
*   **📦 Comprehensive Hardware Monitoring**: Monitor CPU, GPU (Nvidia, Ascend, etc.), memory, and disk usage for performance analysis.
*   **🆚 Experiment Comparison**: Easily compare and analyze multiple experiments to identify the most effective configurations.
*   **🔌 Extensible with Plugins**: Enhance SwanLab's functionality with plugins for notifications, data logging, and custom integrations.

**Key Highlights:**

*   **Streamlined Workflow**: Easily integrate SwanLab into your existing ML workflows.
*   **Intuitive Interface**: A clean and user-friendly interface for easy experiment management and analysis.
*   **Enhanced Collaboration**: Share experiments with colleagues and foster collaborative research.

<br/>

## Table of Contents

*   [🌟 Recent Updates](#-最近更新)
*   [👋🏻 What is SwanLab?](#-什么是swanlab)
*   [📃 Online Demo](#-在线演示)
*   [🏁 Quickstart](#-快速开始)
*   [💻 Self-Hosting](#-自托管)
*   [🔥 Real-world examples](#-实战案例)
*   [🎮 Hardware Monitoring](#-硬件记录)
*   [🚗 Framework Integrations](#-框架集成)
*   [🔌 Plugins and API](#-插件与api)
*   [🆚 Comparison with Similar Tools](#-与熟悉的工具的比较)
*   [👥 Community](#-社区)
*   [📃 License](#-协议)
*   [Star History](#star-history)
<br/>

## 🌟 Recent Updates
- 2025.08.19：🤔更强大的图表渲染性能与低侵入式加载动画，让研究者更聚焦于实验分析本身；集成优秀的[MLX-LM](https://github.com/ml-explore/mlx-lm)、[SpecForge](https://github.com/sgl-project/SpecForge)框架，提供更多场景的训练体验；

- 2025.08.06：👥**训练轻协作**上线，支持邀请项目协作者，分享项目链接与二维码；工作区支持列表视图，支持显示项目Tags；

- 2025.07.29：🚀侧边栏支持**实验筛选、排序**；📊表格视图上线**列控制面板**，能够方便地实现列的隐藏与显示；🔐**多API Key**管理上线，让你的数据更安全；swanlab sync提高了对日志文件完整性的兼容，适配训练崩溃等场景；新图表-PR曲线、ROC曲线、混淆矩阵上线，[文档](https://docs.swanlab.cn/api/py-pr_curve.html)；

- 2025.07.17：📊更强大的**折线图配置**，支持灵活配置线型、颜色、粗细、网格、图例位置等；📹支持**swanlab.Video**数据类型，支持记录与可视化GIF格式文件；全局图表仪表盘支持配置Y轴与最大显示实验数；

- 2025.07.10：📚更强大的**文本视图**，支持Markdown渲染与方向键切换，可由`swanlab.echarts.table`与`swanlab.Text`创建，[Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)

- 2025.07.06：🚄支持**resume断点续训**；新插件**文件记录器**；集成[ray](https://github.com/ray-project/ray)框架，[文档](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html)；集成[ROLL](https://github.com/volcengine/ROLL)框架，感谢[@PanAndy](https://github.com/PanAndy)，[文档](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

- 2025.06.27：📊支持**小折线图局部放大**；支持配置**单个折线图平滑**；大幅改进了图像图表放大后的交互效果；

- 2025.06.20：🤗集成[accelerate](https://github.com/huggingface/accelerate)框架，[PR](https://github.com/huggingface/accelerate/pull/3605)，[文档](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html)，增强分布式训练中的实验记录体验；

- 2025.06.18：🐜集成[AREAL](https://github.com/inclusionAI/AReaL)框架，感谢[@xichengpro](https://github.com/xichengpro)，[PR](https://github.com/inclusionAI/AReaL/pull/98)，[文档](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process)；🖱支持鼠标Hover到侧边栏实验时，高亮相应曲线；支持跨组对比折线图；支持设置实验名裁剪规则；

<details><summary>完整更新日志</summary>

- 2025.06.11：📊支持 **swanlab.echarts.table** 数据类型，支持纯文本图表展示；支持对分组进行**拉伸交互**，以增大同时显示的图表数量；表格视图增加 **指标最大/最小值** 选项；

- 2025.06.08：♻️支持在本地存储完整的实验日志文件，通过 **swanlab sync** 上传本地日志文件到云端/私有化部署端；硬件监控支持**海光DCU**；

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

## 👋🏻 What is SwanLab?

SwanLab is an open-source tool designed to streamline the process of tracking, visualizing, and collaborating on machine learning experiments. With a user-friendly Python API and a clean UI, SwanLab offers features such as:

*   **Visualized Training**: Visualize experiments and analyze the metrics to monitor the training procedure and observe the improvements of your model.
*   **Automatic logging**: Record hyperparameters, model metrics, model architecture, and other meta data automatically without complicated code changes
*   **Experiment comparison**: Compare the results from multiple experiments to discover the inspiration, and accelerate the speed of model iteration.
*   **Teamwork and online sharing**: Improve team collaboration and communication efficiency.

<br/>
<br/>

## 📃 Online Demo

Explore the capabilities of SwanLab with these interactive demos:

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Tracks the training of a simple ResNet50 model on a cat/dog image classification task. |  Tracks the training of Yolov8 on the COCO128 dataset for object detection. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Tracks the instruction fine-tuning of Qwen2 language model, demonstrating the model's ability to follow instructions. | Uses a simple LSTM model to predict Google stock prices, based on the provided dataset. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Provides a progressive experiment on audio classification with ResNet family from ResNet to ResNeXt  | Fine-tuning the Lora based on Qwen2-VL multi-modal model on COCO2014 dataset |

| [EasyR1 Multi-Modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Utilizing EasyR1 framework for multi-modal LLM RL training  | GRPO Training on Qwen2.5-0.5B model for GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)
<br/>
<br/>

## 🏁 Quickstart

Get started with SwanLab in just a few steps:

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

For the latest features, install from source:

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

### 2. Login and Obtain API Key

1.  [Register for a free account](https://swanlab.cn).

2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).

3.  Open your terminal and enter:

```bash
swanlab login
```

Enter your API Key when prompted and press Enter to complete login.

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

That's it! Head to [SwanLab](https://swanlab.cn) to view your first experiment.

<br/>
<br/>

## 💻 Self-Hosting

The self-hosted community version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploying the Self-Hosted Version Using Docker

For details, see: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Quick Install for China Regions:

```bash
./install.sh
```

Install by Pulling the Image from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Directing Experiments to the Self-Hosted Service

Log in to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Once logged in, all your experiments will be recorded to the self-hosted service.

<br/>
<br/>

## 🔥 Real-world Examples

Here are some resources using SwanLab:

**Excellent Tutorial Open Source Projects**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): From zero to large language model principles and practice tutorials ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm): "Open Source Large Model Cookbook" is tailored for Chinese developers, providing a guide to quickly fine-tuning (full parameter/Lora), deploying domestic and international open source large models (LLM) / multi-modal large models (MLLM) in a Linux environment ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, expansion, and reproduction ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
*   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): Concat the visual header of SmolVLM2 with Qwen3-0.6B model for fine-tuning ![GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)

**Outstanding papers using SwanLab:**

*   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
*   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
*   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
*   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorial Articles:**

*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cart Pole Inverted Pendulum](https://docs.swanlab.cn/examples/dqn_cartpole.html)
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
*   [GLM-4.5-Air-LoRA and SwanLab Visualization Record](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
*   [How to do RAG? The Open Source Solution with SwanLab Document Assistant](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟If you would like to include a tutorial, PR is welcomed!

<br/>
<br/>

## 🎮 Hardware Monitoring

SwanLab tracks hardware information and resource usage during AI training. Here's the current support:

| Hardware          | Information Recording | Resource Monitoring | Script                                                                     |
| ----------------- | --------------------- | ------------------- | -------------------------------------------------------------------------- |
| Nvidia GPU        | ✅                     | ✅                   | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)  |
| Ascend NPU        | ✅                     | ✅                   | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)  |
| Apple SOC         | ✅                     | ✅                   | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)  |
| Cambricon MLU     | ✅                     | ✅                   | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU     | ✅                     | ✅                   | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅                     | ✅                   | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU         | ✅                     | ✅                   | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py)  |
| Hygon DCU         | ✅                     | ✅                   | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py)  |
| CPU               | ✅                     | ✅                   | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)         |
| Memory            | ✅                     | ✅                   | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)      |
| Disk              | ✅                     | ✅                   | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)        |
| Network           | ✅                     | ✅                   | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)      |

Please feel free to submit an Issue and PR if you want to record other hardware.

<br/>
<br/>

## 🚗 Framework Integrations

Combine your favorite frameworks with SwanLab! Below is a list of the frameworks we've integrated with. Welcome to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to suggest frameworks for integration.

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

**Other Frameworks**

*   [Tensorboard](https://docs.swanlab.cn/guide_cloud/integration/integration-tensorboard.html)
*   [Weights&Biases](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html)
*   [MLFlow](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html)
*   [HuggingFace Accelerate](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html)
*   [Ray](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html)
*   [Unsloth](https://docs.swanlab.cn/guide_cloud/integration/integration-unsloth.html)
*   [Hydra](https://docs.swanlab.cn/guide_cloud/integration/integration-hydra.html)
*   [Omegaconf](https://docs.swanlab.cn/guide_cloud/integration/integration-omegaconf.html)
*   [OpenAI](https://docs.swanlab.cn/guide