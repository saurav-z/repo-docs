<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## SwanLab: Effortlessly Track, Visualize, and Collaborate on Your Deep Learning Experiments

**SwanLab is an open-source, modern tool designed for tracking, visualizing, and collaborating on deep learning training runs.** Easily integrate with 30+ major frameworks and leverage cloud/offline functionality for streamlined experimentation.  [Explore the original repository](https://github.com/SwanHubX/SwanLab).

<br/>

**Key Features:**

*   📊 **Experiment Tracking & Visualization:**  Intuitive UI for visualizing metrics, hyperparameter tracking, and experiment comparison. Includes support for scalar metrics, images, audio, text, videos, 3D point clouds, biochemical molecules, and custom Echarts charts.
*   ⚡️ **Framework Integrations:** Seamless integration with popular frameworks including PyTorch, Hugging Face Transformers, PyTorch Lightning, LLaMA Factory, MMDetection, Ultralytics, PaddleDetection, LightGBM, XGBoost, and many more.
*   💻 **Hardware Monitoring:** Real-time monitoring of CPU, NPU (Ascend), GPU (Nvidia, Metax, Moore Threads, Hygon), MLU (Cambricon), XPU (Kunlunxin), memory, disk, and network usage.
*   📦 **Experiment Management:** Centralized dashboard for managing multiple projects and experiments.
*   🆚 **Experiment Comparison:** Compare hyperparameters and results across experiments.
*   👥 **Online Collaboration:**  Share results, invite collaborators, and discuss findings.
*   ✉️ **Share Results Easily:** Generate shareable URLs for each experiment, perfect for collaboration and documentation.
*   💻 **Self-Hosted Support:** Use SwanLab offline or on your own infrastructure for complete control.
*   🔌 **Plugin Extensibility:** Extend SwanLab's functionality with plugins for notifications (e.g., Slack, Discord, Lark), CSV logging, and more.

<br/>

## 🌟 What's New

*   **[2025.09.12]**
    *   🔢 Support for creating **scalar charts**, for flexible display of experimental metric statistics; Major upgrade to the organization management page, providing more powerful permission control and project management capabilities;

*   **[2025.08.19]**
    *   🤔 More powerful chart rendering performance and low-invasive loading animation, allowing researchers to focus more on the experiment analysis itself; Integrated excellent [MLX-LM](https://github.com/ml-explore/mlx-lm), [SpecForge](https://github.com/sgl-project/SpecForge) frameworks, providing training experiences for more scenarios;

*   **[2025.08.06]**
    *   👥 **Training Light Collaboration** is online, supporting inviting project collaborators, sharing project links and QR codes; The workspace supports list view and displays project Tags;

*   **[2025.07.29]**
    *   🚀 Sidebar supports **experiment filtering and sorting**; 📊 Table view is online with the **column control panel**, which can conveniently hide and display columns; 🔐 **Multi-API Key** management is online, making your data more secure; swanlab sync improves the compatibility of log file integrity, adapting to scenarios such as training crashes; new charts - PR curve, ROC curve, confusion matrix are online, [documentation](https://docs.swanlab.cn/api/py-pr_curve.html);

*   **[2025.07.17]**
    *   📊 More powerful **line chart configuration**, supporting flexible configuration of line type, color, thickness, grid, legend position, etc.; 📹 Support for **swanlab.Video** data type, supporting recording and visualizing GIF format files; The global chart dashboard supports configuring the Y-axis and the maximum number of displayed experiments;

*   **[2025.07.10]**
    *   📚 More powerful **text view**, supporting Markdown rendering and arrow key switching, can be created by `swanlab.echarts.table` and `swanlab.Text`, [Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)

*   **[2025.07.06]**
    *   🚄 Support **resume breakpoint training**; New plugin **file recorder**; Integrated [ray](https://github.com/ray-project/ray) framework, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); Integrated [ROLL](https://github.com/volcengine/ROLL) framework, thanks to [@PanAndy](https://github.com/PanAndy), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

*   **[2025.06.27]**
    *   📊 Support **local zoom in for small line charts**; Support configuration for **smoothing a single line chart**; Significantly improved the interactive effects after zooming in on image charts;

*   **[2025.06.20]**
    *   🤗 Integrated [accelerate](https://github.com/huggingface/accelerate) framework, [PR](https://github.com/huggingface/accelerate/pull/3605), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html), enhancing the experiment recording experience in distributed training;

<details><summary>Complete Changelog</summary>

*   **[2025.06.18]**
    *   🐜 Integrated [AREAL](https://github.com/inclusionAI/AReaL) framework, thanks to [@xichengpro](https://github.com/xichengpro), [PR](https://github.com/inclusionAI/AReaL/pull/98), [documentation](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process); 🖱 Support highlighting corresponding curves when the mouse hovers over the sidebar experiments; Support cross-group comparison of line charts; Support setting the experiment name cropping rule;

*   **[2025.06.11]**
    *   📊 Supports **swanlab.echarts.table** data type, supports pure text chart display; Supports **interactive stretching** for groups to increase the number of charts displayed simultaneously; The table view adds a **maximum/minimum indicator** option;

*   **[2025.06.08]**
    *   ♻️ Supports storing complete experiment log files locally, uploading local log files to the cloud/private deployment side through **swanlab sync**; Hardware monitoring supports **Hygon DCU**;

*   **[2025.06.01]**
    *   🏸 Supports **free chart dragging**; Supports **ECharts custom charts**, adding 20+ chart types including bar charts, pie charts, and histograms; Hardware monitoring supports **MX GPU**; Integrated **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** framework;

*   **[2025.05.25]**
    *   The log supports recording the **standard error stream**, and the printing information of frameworks such as PyTorch Lightning can be better recorded; Hardware monitoring supports **Moore Threads**; Added the secure protection function of running command records, and the API Key will be automatically hidden;

*   **[2025.05.14]**
    *   Support **experiment Tag**; Support line chart **Log Scale**; Support **group dragging**; Greatly optimized the experience of uploading a large number of indicators; Added `swanlab.OpenApi` open interface;

*   **[2025.05.09]**
    *   Support **line chart creation**; Configuration chart function adds **data source selection** function, supports displaying different indicators in a single chart; Support generating **training project GitHub badge**;

*   **[2025.04.23]**
    *   Support line chart **editing**, support free configuration of the X and Y axis data range and title style of the chart; Chart search supports **regular expressions**; Support hardware detection and monitoring of **Kunlunxin XPU**;

*   **[2025.04.11]**
    *   Support line chart **local area selection**; Support global selection of the step range of the dashboard line chart; Support one-click hiding of all charts;

*   **[2025.04.08]**
    *   Support **swanlab.Molecule** data type, support recording and visualizing biochemical molecular data; Support saving the sorting, filtering, and column order change status in the table view;

*   **[2025.04.07]**
    *   We have completed joint integration with [EvalScope](https://github.com/ModelScope/EvalScope), and now you can use SwanLab to **evaluate large model performance** in EvalScope;

*   **[2025.03.30]**
    *   Support **swanlab.Settings** method, support more refined experiment behavior control; Support **Cambricon MLU** hardware monitoring; Support [Slack notifications](https://docs.swanlab.cn/plugin/notification-slack.html), [Discord notifications](https://docs.swanlab.cn/plugin/notification-discord.html);

*   **[2025.03.21]**
    *   🎉🤗HuggingFace Transformers has officially integrated SwanLab (>=4.50.0 version), [#36433](https://github.com/huggingface/transformers/pull/36433); Added **Object3D chart**, supports recording and visualizing 3D point cloud, [documentation](https://docs.swanlab.cn/api/py-object3d.html); Hardware monitoring supported the recording of GPU video memory (MB), disk utilization, and network up and down traffic;

*   **[2025.03.12]**
    *   🎉🎉SwanLab **private deployment version** has been released!! [🔗Deployment Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab now supports plugin extensions, such as [email notifications](https://docs.swanlab.cn/plugin/notification-email.html), [Feishu notifications](https://docs.swanlab.cn/plugin/notification-lark.html)

*   **[2025.03.09]**
    *   Support **experiment sidebar widening**; Added an external display Git code button; Added **sync_mlflow** function, support synchronizing experiment tracking with mlflow framework;

*   **[2025.03.06]**
    *   We have completed joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio), and now you can use SwanLab in DiffSynth Studio to **track and visualize Diffusion model text-to-image/video experiments**, [usage instructions](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html);

*   **[2025.03.04]**
    *   Added the **MLFlow conversion** function, supports converting MLFlow experiments to SwanLab experiments, [usage instructions](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html);

*   **[2025.03.01]**
    *   Added the **Move Experiment** function, you can now move the experiment to different projects of different organizations;

*   **[2025.02.24]**
    *   We have completed joint integration with [EasyR1](https://github.com/hiyouga/EasyR1), and now you can use SwanLab in EasyR1 to **track and visualize multi-modal large model reinforcement learning experiments**, [usage instructions](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)

*   **[2025.02.18]**
    *   We have completed joint integration with [Swift](https://github.com/modelscope/ms-swift), and now you can use SwanLab in Swift's CLI/WebUI to **track and visualize large model fine-tuning experiments**, [usage instructions](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html).

*   **[2025.02.16]**
    *   Added the **Chart Move Grouping, Create Grouping** function.

*   **[2025.02.09]**
    *   We have completed joint integration with [veRL](https://github.com/volcengine/verl), and now you can use SwanLab in veRL to **track and visualize large model reinforcement learning experiments**, [usage instructions](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html).

*   **[2025.02.05]**
    *   `swanlab.log` supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), adapting to Jax framework features; supports `name` and `notes` parameters;

*   **[2025.01.22]**
    *   Added `sync_tensorboardX` and `sync_tensorboard_torch` functions to support the synchronization of experimental tracking with these two TensorBoard frameworks;

*   **[2025.01.17]**
    *   Added `sync_wandb` function, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html), supports synchronization with Weights & Biases experimental tracking; Significantly improved log rendering performance

*   **[2025.01.11]**
    *   The cloud version has greatly optimized the performance of the project table and supports drag and drop, sorting, filtering, and other interactions

*   **[2025.01.01]**
    *   Added line chart **persistent smoothing**, drag-and-drop to change the size of line charts, optimize the chart browsing experience

*   **[2024.12.22]**
    *   We have completed joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), and now you can use SwanLab in LLaMA Factory to **track and visualize large model fine-tuning experiments**, [usage instructions](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger).

*   **[2024.12.15]**
    *   The **Hardware Monitoring (0.4.0)** function is online, supporting the recording and monitoring of system-level information of CPU, NPU (Ascend), and GPU (Nvidia).

*   **[2024.12.06]**
    *   Added the integration of [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html) and [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html); Increased the limit on the length of a single line of log recording.

*   **[2024.11.26]**
    *   The environment tab-hardware section supports recognizing **Huawei Ascend NPU** and **Kunpeng CPU**; The cloud vendor section supports recognizing Qingyun **Jishizhisuan**.

</details>

<br>

## 🏁 Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Code Installation</summary>

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

1.  [Register an account](https://swanlab.cn) (free)
2.  Log in and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and run:

```bash
swanlab login
```

Enter your API Key when prompted, and press Enter to log in.

### 3. Integrate SwanLab into your code

```python
import swanlab

# Initialize a new SwanLab experiment
swanlab.init(
    project="my-first-ml",
    config={'learning-rate': 0.003},
)

# Record metrics
for i in range(10):
    swanlab.log({"loss": i, "acc": i})
```

Now, head over to [SwanLab](https://swanlab.cn) to view your first experiment!

<br>

## 💻 Self-Hosting

Self-hosting the community version allows you to view the SwanLab dashboard offline.  See the documentation for setup instructions.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy the self-hosted version using Docker

For details, refer to: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For fast installation in China:

```bash
./install.sh
```

To install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Direct Experiments to Self-Hosted Service

Log in to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, you can record experiments to the self-hosted service.

<br>

## 🔥 Real-World Examples

**Excellent Open Source Projects Using SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): A tutorial on the principles and practices of large language models from scratch ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm):  A guide to open-source large models tailored for Chinese users, featuring fast fine-tuning (full parameter/Lora) and deployment of domestic and international open-source large models (LLM) / multi-modal large models (MLLM) ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): Interpretations, extensions, and reproductions of the DeepSeek series of work ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
*   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): Fine-tuning of the visual head of SmolVLM2 and the Qwen3-0.6B model ![GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)
*   [OPPO/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models): End-to-end Agent foundation model through multi-Agent distillation and Agent RL. ![GitHub Repo stars](https://img.shields.io/github/stars/OPPO-PersonalAI/Agent_Foundation_Models)

**Excellent Papers Using SwanLab:**

*   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
*   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
*   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
*   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorial Articles:**
*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat and Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multi-modal Large Model Fine-tuning Practical](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multi-modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA and SwanLab Visualized Record](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
*   [How to do RAG? SwanLab document assistant plan is open source](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟 If you have a tutorial you would like to include, please submit a PR!

<br>

## 🎮 Hardware Monitoring

SwanLab records hardware information and resource usage during AI training. Supported hardware is as follows:

| Hardware | Information Recording | Resource Monitoring | Script |
| --- | --- | --- | --- |
| Nvidia GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
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

If you would like to record other hardware, please submit an Issue and PR!

<br>

## 🚗 Framework Integrations

Integrate your favorite frameworks with SwanLab!  
Below is the list of frameworks we have integrated.  Please submit an [Issue](https://github.com/swanhubx/swanlab/issues) to give us feedback on the framework you want us to integrate.

**Base Frameworks**
* [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
* [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
* [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Fine-tuning Frameworks**
* [PyTorch Lightning](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-lightning.html)
* [HuggingFace Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-transformers.html)
* [LLaMA Factory](https://docs.swanlab.cn/guide_cloud/integration/integration-llama-factory.html)
* [Modelscope Swift](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html)
* [DiffSynth Studio](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html)
* [Sentence Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-sentence-transformers.html)
* [PaddleNLP](https://docs.swanlab.cn/guide_cloud/integration/integration-paddlenlp.html)
* [OpenMind](https://modelers.cn/docs/zh/openmind-library/1.0.0/basic_tutorial/finetune/finetune_pt.html#%E8%AE%AD%E7%BB%83%E7%9B%91%E6%8E%A7)
* [Torchtune](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-torchtune.html)
* [XTuner](https://docs.swanlab.cn/guide_cloud/integration/integration-xtuner.html)
* [MMEngine](https://docs.swanlab.cn/guide_cloud/integration/integration-mmengine.html)
* [FastAI](https://docs.swanlab.cn/guide_cloud/integration/integration-fastai.html)
* [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html)
* [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html)
* [MLX-LM](https://docs.swanlab.cn/guide_cloud/integration/integration-mlx-lm.html)

**Evaluation Frameworks**
* [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)

**Computer Vision**
* [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
* [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
* [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
* [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
* [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)

**Reinforcement Learning**
* [Stable Baseline3](https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html)
* [veRL](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)
* [HuggingFace trl](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-trl.html)
* [EasyR1](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
* [AReaL](https://docs.swanlab.cn/guide_cloud/integration/integration-areal.html)
* [ROLL](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

**Other Frameworks:**
* [Tensorboard](https://docs.swanlab.cn/guide_cloud/integration/integration-tensorboard.html)
* [Weights&Biases](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html)
* [MLFlow](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html)
* [HuggingFace Accelerate](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html)
* [Ray](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html)
* [Unsloth](https://docs.swanlab.cn/guide_cloud/integration/integration-unsloth.html)
* [Hydra](https://docs.swanlab.cn/guide_cloud/integration/integration-hydra.html)
* [Omegaconf](https://docs.swanlab.cn/guide_cloud/integration/integration-omegaconf.html)
* [OpenAI](https://docs.swanlab.cn/guide_cloud/integration/integration-openai.html)
* [ZhipuAI](https://docs.swanlab.cn/guide_cloud/integration/integration-zhipuai.html)

[More Integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br>

## 🔌 Plugins & API

Extend the capabilities of SwanLab with plugins and enhance your experiment management experience!

*   [Customize Your Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notification](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notification](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notification](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeCom Notification](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notification](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notification](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [Filelogdir Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparisons to Similar Tools

### Tensorboard vs SwanLab

*   **☁️ Online Support**: SwanLab makes it easy to synchronize and save your training experiments online in the cloud, which is convenient for remotely viewing the progress of training, managing historical projects, sharing experiment links, sending real-time message notifications, viewing experiments on multiple devices, and more. Tensorboard is an offline experiment tracking tool.
*   **👥 Multi-Person Collaboration**: When performing multi-person, cross-team machine learning collaboration, SwanLab makes it easy to manage multi-person training projects, share experiment links, and communicate and discuss across spaces. Tensorboard is designed primarily for individuals and is difficult for multi-person collaboration and sharing of experiments.
*   **💻 Persistent, Centralized Dashboard**: No matter where you train your model, whether on a local computer, in a lab cluster, or in a public cloud GPU instance, your results are recorded in the same centralized dashboard. Using TensorBoard, it takes time to copy and manage TFEvent files from different machines.
*   **💪 More Powerful Tables**: