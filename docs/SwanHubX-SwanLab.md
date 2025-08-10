<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br/>

## SwanLab: The Open-Source Deep Learning Experiment Tracking and Visualization Tool

**SwanLab** empowers researchers with an intuitive platform for tracking, visualizing, and collaborating on deep learning experiments, whether you're training locally or in the cloud.  <a href="https://github.com/SwanHubX/SwanLab">Explore the code on GitHub!</a>

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/SwanHubX/SwanLab/releases)
[![Docker Image](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

![](readme_files/swanlab-overview.png)

[中文 / English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking:** Easily log metrics, hyperparameters, and metadata for comprehensive experiment analysis.
*   **Rich Visualization:** Interactive dashboards and charts for visualizing training progress, including line plots, media, 3D point clouds, and custom ECharts.
*   **Framework Integrations:** Seamlessly integrates with over 30 popular deep learning frameworks, including PyTorch, TensorFlow, Hugging Face, and more.
*   **Hardware Monitoring:** Monitor CPU, GPU (Nvidia, Ascend, others), memory, disk, and network usage during training.
*   **Experiment Management:** Organize and compare experiments with a centralized dashboard and intuitive filtering.
*   **Collaboration:** Share experiments and collaborate with your team in real-time.
*   **Self-Hosting:** Run SwanLab locally or on your own servers for complete control over your data.
*   **Plugin Ecosystem:** Extend functionality with plugins for notifications, data writing, and more.
*   **Metadata Support:** Support logging of scalar metrics, images, audio, text, video, 3D point clouds, biochemical molecules, and Echarts custom charts.
*   **Chart Types**: Support for line charts, media charts (images, audio, text, video), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, and [custom charts](https://docs.swanlab.cn/guide_cloud/experiment_track/log-custom-chart.html)...
*   **LLM Content Visualization Components**: Built for large language model training scenarios, it offers text content visualization charts and supports Markdown rendering.

<br>

## Recent Updates

*   **2025.08.06**: 👥**Training Collaboration** launched, supporting project collaborators, sharing project links and QR codes; Workspace supports list view, shows project Tags;
*   **2025.07.29**: 🚀Sidebar supports **Experiment Filtering and Sorting**; 📊Table view launched **Column Control Panel**, allows easy column hiding and display; 🔐**Multi-API Key** management launched, making your data more secure; swanlab sync improves compatibility with log file integrity, adapting to training crashes; New charts-PR curve, ROC curve, confusion matrix launched, [Documentation](https://docs.swanlab.cn/api/py-pr_curve.html);
*   **2025.07.17**: 📊More powerful **line chart configuration**, supports flexible configuration of line type, color, thickness, grid, legend position, etc.; 📹Supports **swanlab.Video** data type, supports recording and visualizing GIF format files; Global chart dashboard supports configuring the Y-axis and the maximum number of displayed experiments;
*   **2025.07.10**: 📚More powerful **text view**, supports Markdown rendering and arrow key switching, can be created by `swanlab.echarts.table` and `swanlab.Text`, [Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)
*   **2025.07.06**: 🚄Supports **resume breakpoint training**; New plugin **File Logger**; Integrated [ray](https://github.com/ray-project/ray) framework, [Documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); Integrated [ROLL](https://github.com/volcengine/ROLL) framework, thanks to [@PanAndy](https://github.com/PanAndy), [Documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)
*   **2025.06.27**: 📊Supports **zoom in on small line charts**; Supports configuration of **single line chart smoothing**; Significantly improved the interactive effect after image chart magnification;
*   **2025.06.20**: 🤗Integrated [accelerate](https://github.com/huggingface/accelerate) framework, [PR](https://github.com/huggingface/accelerate/pull/3605), [Documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html), enhancing the experiment logging experience in distributed training;
*   **2025.06.18**: 🐜Integrated [AREAL](https://github.com/inclusionAI/AReaL) framework, thanks to [@xichengpro](https://github.com/xichengpro), [PR](https://github.com/inclusionAI/AReaL/pull/98), [Documentation](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process); 🖱Supports highlighting the corresponding curve when hovering the mouse over the sidebar experiment; Supports cross-group comparison of line charts; Supports setting experiment name clipping rules;
*   **2025.06.11**: 📊Supports **swanlab.echarts.table** data type, supports pure text chart display; Supports **stretching interaction** for grouping to increase the number of charts displayed simultaneously; The table view adds **indicator maximum/minimum** options;

<details><summary>Full Update Log</summary>

- 2025.06.08: ♻️Supports storing complete experiment log files locally, and uploads local log files to the cloud/private deployment end through **swanlab sync**; Hardware monitoring supports **Hygon DCU**;
- 2025.06.01: 🏸Supports **chart free dragging**; Supports **ECharts custom charts**, adding 20+ chart types including bar charts, pie charts, and histograms; Hardware monitoring supports **Moore Threads GPU**; Integrated **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** framework;
- 2025.05.25: The log supports recording **standard error streams**, and the print information of frameworks such as PyTorch Lightning can be better recorded; Hardware monitoring supports **Moore Threads**; Added run command recording security protection function, API Key will be automatically hidden;
- 2025.05.14: Supports **experiment Tag**; Supports line chart **Log Scale**; Supports **group dragging**; Significantly optimizes the experience of uploading a large number of indicators; Added `swanlab.OpenApi` open interface;
- 2025.05.09: Supports **line chart creation**; The configure chart function adds the **data source selection** function, which supports displaying different indicators in a single chart; Supports generating **training project GitHub badges**;
- 2025.04.23: Supports line chart **editing**, supports freely configuring the X and Y axis data range and title style of the chart; Chart search supports **regular expressions**; Supports hardware detection and monitoring of **Kunlunxin XPU**;
- 2025.04.11: Supports line chart **local area selection**; Supports global selection of the step range of the dashboard line chart; Supports one-click hiding of all charts;
- 2025.04.08: Supports the **swanlab.Molecule** data type, supports recording and visualizing biochemical molecular data; Supports saving the sorting, filtering, and column order change status in the table view;
- 2025.04.07: We have completed the joint integration with [EvalScope](https://github.com/ModelScope/EvalScope), now you can use SwanLab in EvalScope to **evaluate large model performance**;
- 2025.03.30: Supports the **swanlab.Settings** method, supports more refined experiment behavior control; Supports **Cambricon MLU** hardware monitoring; Supports [Slack notification](https://docs.swanlab.cn/plugin/notification-slack.html), [Discord notification](https://docs.swanlab.cn/plugin/notification-discord.html);
- 2025.03.21: 🎉🤗HuggingFace Transformers has officially integrated SwanLab (>=4.50.0 version), [#36433](https://github.com/huggingface/transformers/pull/36433); Added the **Object3D chart**, supports recording and visualizing 3D point cloud, [Documentation](https://docs.swanlab.cn/api/py-object3d.html); Hardware monitoring supports the recording of GPU video memory (MB), disk utilization, and network traffic;
- 2025.03.12: 🎉🎉SwanLab **private deployment version** has been released!! [🔗Deployment Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab now supports plugin extensions, such as [Email notification](https://docs.swanlab.cn/plugin/notification-email.html), [Feishu notification](https://docs.swanlab.cn/plugin/notification-lark.html)
- 2025.03.09: Supports **experiment sidebar widening**; Added the external Git code button; Added the **sync_mlflow** function, supports synchronizing experiment tracking with the mlflow framework;
- 2025.03.06: We have completed the joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio), now you can use SwanLab in DiffSynth Studio to **track and visualize Diffusion model text-to-image/video experiments**, [User Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html);
- 2025.03.04: Added the **MLFlow conversion** function, supports converting MLFlow experiments to SwanLab experiments, [User Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html);
- 2025.03.01: Added the **Move Experiment** function, now you can move experiments to different projects of different organizations;
- 2025.02.24: We have completed the joint integration with [EasyR1](https://github.com/hiyouga/EasyR1), now you can use SwanLab in EasyR1 to **track and visualize multi-modal large model reinforcement learning experiments**, [User Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
- 2025.02.18: We have completed the joint integration with [Swift](https://github.com/modelscope/ms-swift), now you can use SwanLab in Swift's CLI/WebUI to **track and visualize large model fine-tuning experiments**, [User Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html).
- 2025.02.16: Added the **Chart Move Grouping, Create Grouping** function.
- 2025.02.09: We have completed the joint integration with [veRL](https://github.com/volcengine/verl), now you can use SwanLab in veRL to **track and visualize large model reinforcement learning experiments**, [User Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html).
- 2025.02.05: `swanlab.log` supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), adapts to Jax framework characteristics; Supports `name` and `notes` parameters;
- 2025.01.22: Added the `sync_tensorboardX` and `sync_tensorboard_torch` functions, supports synchronizing experiment tracking with these two TensorBoard frameworks;
- 2025.01.17: Added the `sync_wandb` function, [Documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html), supports synchronizing experiment tracking with Weights & Biases; Significantly improved the log rendering performance
- 2025.01.11: The cloud version has significantly optimized the performance of the project table, and supports interactions such as drag and drop, sorting, and filtering
- 2025.01.01: Added line chart **persistent smoothing**, line chart drag to change size, optimize chart browsing experience
- 2024.12.22: We have completed the joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), now you can use SwanLab in LLaMA Factory to **track and visualize large model fine-tuning experiments**, [User Guide](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger).
- 2024.12.15: **Hardware monitoring (0.4.0)** function launched, supporting the recording and monitoring of system-level information of CPU, NPU (Ascend), GPU (Nvidia).
- 2024.12.06: Added the integration of [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html), [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html); Increased the limit on the length of a single line of log recording.
- 2024.11.26: The environment tab-hardware section supports identifying **Huawei Ascend NPU** and **Kunpeng CPU**; The cloud vendor section supports identifying Qingyun **Jishi Zhisuan**.

</details>

<br>

## Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

For the latest features, install from source:

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Install Offline Dashboard Extension</summary>

[Offline Dashboard Documentation](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get Your API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Log in and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and run:

```bash
swanlab login
```

Enter your API Key when prompted, then press Enter to log in.

### 3. Integrate SwanLab into Your Code

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

Now, visit [SwanLab](https://swanlab.cn) to view your experiment!

<br/>

## Self-Hosting

The self-hosted community version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted Version using Docker

See the full instructions: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For China:

```bash
./install.sh
```

To pull the image from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point Your Experiments to the Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, your experiments will be recorded to the self-hosted service.

<br>

## Example Projects

**Awesome Tutorials & Open Source Projects Using SwanLab:**
- [happy-llm](https://github.com/datawhalechina/happy-llm): A tutorial on the principle and practice of large language models from scratch ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
- [self-llm](https://github.com/datawhalechina/self-llm): "Open Source Large Model Handbook" custom tailored for Chinese users, providing a guide to rapidly fine-tune (full parameters/Lora) and deploy domestic and international open source large models (LLM)/multi-modal large models (MLLM) on Linux. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
- [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, expansion and reproduction.![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)

**Papers Using SwanLab:**
- [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
- [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
- [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)

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
- [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
- [LLM Pre-training](https://docs.swanlab.cn/examples/pretrain_llm.html)
- [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
- [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
- [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
- [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
- [Qwen2-VL Multi-Modal Large Model Fine-tuning Practice](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
- [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
- [Qwen3-SmVL-0.6B Multi-Modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
- [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
- [GLM-4.5-Air-LoRA and SwanLab Visualization Records](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)

🌟 Submit a PR with your tutorial!

<br>

## Hardware Monitoring

SwanLab tracks **hardware information** and **resource usage** during your AI training runs.

| Hardware     | Information Recording | Resource Monitoring | Script                                                            |
| :----------- | :-------------------: | :-----------------: | :---------------------------------------------------------------- |
| Nvidia GPU   |          ✅          |         ✅         | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)   |
| Ascend NPU   |          ✅          |         ✅         | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)   |
| Apple SOC    |          ✅          |         ✅         | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)    |
| Cambricon MLU |          ✅          |         ✅         | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU|          ✅          |         ✅         | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU          |          ✅          |         ✅         | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)           |
| Memory       |          ✅          |         ✅         | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)        |
| Disk         |          ✅          |         ✅         | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)          |
| Network      |          ✅          |         ✅         | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)      |

If you want to record other hardware, submit an Issue or PR!

<br>

## Framework Integrations

Integrate your favorite frameworks with SwanLab!

Here is a list of integrated frameworks. Feel free to provide feedback on the frameworks you want to integrate by submitting an [Issue](https://github.com/swanhubx/swanlab/issues).

**Core Frameworks**
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

## Plugin and API

Extend SwanLab's capabilities with plugins!

-   [Customize Your Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
-   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [WeCom Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [File Logger](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:
- [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparison: Tensorboard vs. Weights