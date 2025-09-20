<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br>

## 🚀 Supercharge Your Deep Learning Experiments with SwanLab

**SwanLab is an open-source, user-friendly tool for tracking, visualizing, and collaborating on your deep learning experiments, seamlessly integrating with 30+ popular frameworks.**  Unlock deeper insights, accelerate your research, and streamline your workflow with an intuitive interface and powerful features.  [Explore the SwanLab Repository](https://github.com/SwanHubX/SwanLab).

<br/>

**Key Features:**

*   📊 **Experiment Tracking & Visualization:**
    *   Track crucial metrics, hyperparameters, and model performance.
    *   Visualize training progress with intuitive charts (line plots, images, audio, text, videos, 3D point clouds, custom Echarts).
    *   View LLM-generated content with custom visualizations.
    *   Auto-logging of experiment details like Git repo, hardware environment, and Python environment.
    *   Support for resuming interrupted training runs.

*   ⚡️ **Framework Integrations:**
    *   Seamless integration with 30+ popular frameworks: PyTorch, Hugging Face Transformers, PyTorch Lightning, LLaMA Factory, MMDetection, Ultralytics, PaddleDetetion, LightGBM, XGBoost, and many more.

*   💻 **Hardware Monitoring:**
    *   Real-time monitoring and logging of CPU, NPU (Ascend), GPU (Nvidia, MLU, Kunlunxin, DCU, MetaX, Moore Threads), and memory usage.

*   📦 **Experiment Management:**
    *   Centralized dashboard for managing multiple projects and experiments.
    *   Intuitive UI to filter, sort, and organize experiments.

*   🆚 **Comparative Analysis:**
    *   Compare different experiments through online tables and comparison charts.

*   👥 **Collaboration:**
    *   Collaborative training features for teams to share and discuss experiment results.

*   ✉️ **Result Sharing:**
    *   Share experiments with persistent URLs, ideal for presentations and publications.

*   💻 **Self-Hosting:**
    *   Run SwanLab locally or on your own servers for offline access and data privacy.

*   🔌 **Plugin Extensibility:**
    *   Extend functionality with plugins for notifications (Slack, Discord, etc.) and data writers (CSV, etc.).

<br>

## 🌟 Recent Updates

*   **2025.09.12:** Added support for creating **scalar charts** for flexible display of experiment metrics; improved organizational management with advanced permission controls and project management capabilities.

*   **2025.08.19:** Enhanced chart rendering performance with low-intrusive loading animations; integrated with MLX-LM and SpecForge frameworks to enhance training experiences.

*   **2025.08.06:** Launched **training co-operation** features to enable collaboration on projects with project collaborators with project links and QR codes; enhanced workspace with list views to display project tags.

*   **2025.07.29:** Updated sidebar with **experiment filtering and sorting** functionality; added column control panel to table views for hiding/showing columns; introduced **multi-API Key management** features for data security; improved compatibility of swanlab sync to support training crashes. Added new charts - PR curve, ROC curve, confusion matrix.

*   **2025.07.17:** Introduced more robust **line chart configurations** to flexibly configure line types, colors, thickness, grids, and legend positions; added **swanlab.Video** data type to record and visualize GIF files; updated global chart dashboard to configure Y-axis and maximum displayed experiments.

*   **2025.07.10:** Launched **text view** for Markdown rendering and directional switching, can be created by `swanlab.echarts.table` and `swanlab.Text`.

*   **2025.07.06:** Supported **resume breakpoint continuation training**; launched new plugin **file recorder**; integrated [ray](https://github.com/ray-project/ray) framework.

*   **2025.06.27:** Supported **zoom for small line charts**; supported the configuration of **smooth single line charts**; significantly improved the interactive effect after zooming in image charts.

*   **2025.06.20:** Integrated the [accelerate](https://github.com/huggingface/accelerate) framework to enhance the experimental recording experience in distributed training.

*(See the full changelog in the original README for more details.)*

<details><summary>Complete Update Log</summary>

-   2025.06.18：🐜Integrated [AREAL](https://github.com/inclusionAI/AReaL) framework, thanks [@xichengpro](https://github.com/xichengpro), [PR](https://github.com/inclusionAI/AReaL/pull/98), [documentation](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process)；🖱Support highlighting the corresponding curve when the mouse hovers over the sidebar experiment; support cross-group comparison of line charts; support setting experiment name cropping rules;

-   2025.06.11：📊Support **swanlab.echarts.table** data type, support pure text chart display; support **interactive stretching** for groups to increase the number of charts displayed at the same time; table view adds **metric maximum/minimum** options;

-   2025.06.08：♻️Supports storing complete experimental log files locally, and uploads local log files to the cloud/privately deployed end through **swanlab sync**; hardware monitoring supports **Hycon DCU**;

-   2025.06.01：🏸Support **free drag and drop of charts**; support **ECharts custom charts**, adding more than 20 chart types including bar charts, pie charts, and histograms; hardware monitoring supports **MX GPU**; Integrated **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** framework;

-   2025.05.25：The log supports recording **standard error stream**, and the printing information of PyTorch Lightning and other frameworks can be better recorded; hardware monitoring supports **Moore Threads**; added the security protection function for running command records, and the API Key will be automatically hidden;

-   2025.05.14：Support **experimental Tag**; support line chart **Log Scale**; support **group drag and drop**; greatly optimize the experience of uploading a large number of indicators; add`swanlab.OpenApi`open interface;

-   2025.05.09：Support **line chart creation**; the chart configuration function adds the **data source selection** function, supports displaying different indicators for a single chart; support generating **training project GitHub badge**;

-   2025.04.23：Support line chart **editing**, support free configuration of X, Y axis data range and title style of the chart; chart search supports **regular expressions**; supports hardware detection and monitoring of **Kunlunxin XPU**;

-   2025.04.11：Support line chart **local area selection**; support global selection of the step range of the dashboard line chart; support one-click hiding of all charts;

-   2025.04.08：Support **swanlab.Molecule** data type, support recording and visualizing biochemical molecular data; support saving the sorting, filtering, and column order change status in the table view;

-   2025.04.07：We have completed the joint integration with [EvalScope](https://github.com/ModelScope/EvalScope), and now you can use SwanLab in EvalScope to **evaluate the performance of large models**;

-   2025.03.30：Support **swanlab.Settings** method, support more refined experimental behavior control; support **Cambrian MLU** hardware monitoring; support [Slack notifications](https://docs.swanlab.cn/plugin/notification-slack.html), [Discord notifications](https://docs.swanlab.cn/plugin/notification-discord.html);

-   2025.03.21：🎉🤗HuggingFace Transformers has officially integrated SwanLab (>=4.50.0 version), [#36433](https://github.com/huggingface/transformers/pull/36433); added **Object3D chart**, supporting the recording and visualization of three-dimensional point clouds, [documentation](https://docs.swanlab.cn/api/py-object3d.html); hardware monitoring supports the recording of GPU video memory (MB), disk utilization rate, and network uplink and downlink;

-   2025.03.12：🎉🎉SwanLab **private deployment version** has been released! [🔗Deployment Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab now supports plugin extensions, such as [email notifications](https://docs.swanlab.cn/plugin/notification-email.html), [Feishu notifications](https://docs.swanlab.cn/plugin/notification-lark.html)

-   2025.03.09：Support **experiment sidebar widening**; new display Git code button; new **sync_mlflow** function, support synchronizing experimental tracking with the mlflow framework;

-   2025.03.06：We have completed the joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio), and now you can use SwanLab in DiffSynth Studio to **track and visualize Diffusion model text-to-image/video experiments**, [user guide](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html);

-   2025.03.04：Added **MLFlow conversion** function, supporting the conversion of MLFlow experiments to SwanLab experiments, [user guide](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html);

-   2025.03.01：Added the **move experiment** function, and experiments can now be moved to different projects of different organizations;

-   2025.02.24：We have completed the joint integration with [EasyR1](https://github.com/hiyouga/EasyR1), and now you can use SwanLab in EasyR1 to **track and visualize multimodal large model reinforcement learning experiments**, [user guide](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)

-   2025.02.18：We have completed the joint integration with [Swift](https://github.com/modelscope/ms-swift), and now you can use SwanLab in Swift's CLI/WebUI to **track and visualize large model fine-tuning experiments**, [user guide](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html).

-   2025.02.16：Added the **chart move grouping, create grouping** function.

-   2025.02.09：We have completed the joint integration with [veRL](https://github.com/volcengine/verl), and now you can use SwanLab in veRL to **track and visualize large model reinforcement learning experiments**, [user guide](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html).

-   2025.02.05：`swanlab.log` supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), adapting to Jax framework features; support `name` and `notes` parameters;

-   2025.01.22：Added `sync_tensorboardX` and `sync_tensorboard_torch` functions to support synchronizing experimental tracking with these two TensorBoard frameworks;

-   2025.01.17：Added `sync_wandb` function, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html), support for synchronization of experimental tracking with Weights & Biases; greatly improved log rendering performance

-   2025.01.11：The cloud version greatly optimizes the performance of the project table and supports drag-and-drop, sorting, filtering and other interactions

-   2025.01.01：Added line chart **persistent smoothing**, line chart drag-and-drop changing size, and optimized chart browsing experience

-   2024.12.22：We have completed the joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), and now you can use SwanLab in LLaMA Factory to **track and visualize large model fine-tuning experiments**, [user guide](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger).

-   2024.12.15：**Hardware monitoring (0.4.0)** feature is online, supporting the recording and monitoring of system-level information of CPU, NPU (Ascend), and GPU (Nvidia).

-   2024.12.06：Added integration of [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html) and [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html); increased the limit on the length of a single line of log records.

-   2024.11.26：Environment tab - The hardware part supports the identification of **Huawei Ascend NPU** and **Kunpeng CPU**; the cloud vendor part supports the identification of Qingyun **Jishi Zhisuan**.

</details>

<br>

## 📃 Online Demos

Explore SwanLab's capabilities with these interactive demos:

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Track image classification with a simple ResNet50 model on a cat/dog dataset. | Track the hyper-parameters and metrics from training a Yolov8 model on the COCO128 dataset. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Track the fine-tuning of the Qwen2 large language model and complete simple instruction following. | Use a simple LSTM model trained on a Google stock dataset to predict future stock prices. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiment of audio classification from ResNet to ResNeXt | Based on Qwen2-VL multimodal large model, fine-tuning on COCO2014 dataset. |

| [EasyR1 Multimodal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Using the EasyR1 framework for multimodal LLM RL training | Based on the Qwen2.5-0.5B model, GRPO training on the GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quickstart

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

1.  [Register for a free account](https://swanlab.cn)
2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and enter:

```bash
swanlab login
```

When prompted, enter your API Key and press Enter to log in.

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

Congratulations! View your first SwanLab experiment on [SwanLab](https://swanlab.cn).

<br>

## 💻 Self-Hosting

Self-hosting allows you to run SwanLab on your own infrastructure.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy a Self-Hosted Version Using Docker

For detailed instructions, see the [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Quick Installation for China:

```bash
./install.sh
```

Install by pulling images from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point Experiments to Your Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, your experiments will be recorded on the self-hosted service.

<br>

## 🔥 Real-World Examples

**Excellent Tutorial Open Source Projects using SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): Large language model principle and practice tutorial from scratch ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm): "Open Source Large Model Guide" tailored for Chinese users, based on a Linux environment for quick fine-tuning (all parameters/Lora), deploying domestic and foreign open source large models (LLM)/multi-modal large models (MLLM) tutorial ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, expansion and reproduction ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
*   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): Used the SmolVLM2's visual head to connect and fine-tune the Qwen3-0.6B model ![GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)
*   [OPPO/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models): End-to-end agent foundation models through multi-agent distillation and Agent RL. ![GitHub Repo stars](https://img.shields.io/github/stars/OPPO-PersonalAI/Agent_Foundation_Models)

**Excellent Papers Using SwanLab:**

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
*   [LLM Pre-training](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multimodal Large Model Fine-tuning Hands-on Practice](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA and SwanLab Visualization Record](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
*   [How to do RAG? SwanLab document assistant solution is open-sourced](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟  Feel free to submit a PR if you have any tutorials you'd like to include!

<br>

## 🎮 Hardware Monitoring

SwanLab monitors hardware information and resource usage during AI training.

| Hardware | Information Recording | Resource Monitoring | Script |
| --- | --- | --- | --- |
| NVIDIA GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU | ✅ | ✅ | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC | ✅ | ✅ | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambrian MLU | ✅ | ✅ | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅ | ✅ | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU | ✅ | ✅ | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory | ✅ | ✅ | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk | ✅ | ✅ | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

If you want to record other hardware, please submit an Issue and PR!

<br>

## 🚗 Framework Integrations

Integrate your favorite frameworks with SwanLab!

Here is a list of our integrated frameworks, and feel free to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to provide feedback on the framework you want to integrate.

**Basic Frameworks**

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Proprietary/Fine-tuning Frameworks**

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

## 🔌 Plugins & API

Extend SwanLab with plugins and APIs to enhance your experiment management.

*   [Customize your Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Enterprise Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Directory Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open Interfaces:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparisons with Familiar Tools

### TensorBoard vs. SwanLab

*   ☁️ **Online Accessibility:** SwanLab provides convenient online experiment synchronization and storage for easy access to training progress, project management, sharing experiment links, and real-time notifications. TensorBoard operates offline.
*   👥 **Collaboration:**