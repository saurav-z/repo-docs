<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## 🚀 SwanLab: Supercharge Your Deep Learning Experiments with Open-Source Tracking and Visualization

**SwanLab is an open-source, user-friendly tool designed to streamline your deep learning workflow by tracking, visualizing, and collaborating on your experiments, seamlessly integrating with 30+ popular frameworks.** ([Original Repo](https://github.com/SwanHubX/SwanLab))

<br>

**Key Features:**

*   **📊 Experiment Tracking & Visualization:**
    *   Track key metrics, hyperparameters, and model performance effortlessly.
    *   Visualize training progress with intuitive charts and graphs.
    *   Supports various data types: scalar metrics, images, audio, text, videos, 3D point clouds, and custom ECharts visualizations.
    *   Includes out-of-the-box support for various chart types like line charts, media charts, 3D point clouds, and more.
    *   Specialized LLM content visualization component with markdown rendering.
    *   Records: Logs, hardware environment, Git repository information, Python environment, library lists, and project run directories automatically.
    *   Supports resuming interrupted training with data addition.
*   **☁️ Cloud and Offline Support:** Use SwanLab on the cloud (like Weights & Biases) or run experiments locally for full flexibility.
*   **🔌 Extensive Framework Integrations:** Seamlessly integrates with 30+ popular machine learning frameworks.
*   **💻 Hardware Monitoring:** Real-time monitoring of CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (Muxi XPU), Moore Threads GPU, memory, disk, and network usage.
*   **📦 Experiment Management:** Manage multiple projects and experiments through a centralized dashboard.
*   **🆚 Result Comparison:** Compare experiments through online tables and comparison charts, uncovering insights for model iteration.
*   **👥 Collaborative Features:** Collaborate on experiments with your team, view training records, and provide feedback.
*   **✉️ Sharing:** Share results with ease via persistent URLs for each experiment.
*   **🔌 Plugin Extensibility:** Expand SwanLab's functionality with customizable plugins for notification, writing etc.
*   **💻 Self-Hosted Version:** Run and visualize SwanLab's experiments on your infrastructure with the community version.

<br>

## 🌟 Recent Updates

*   **2025.08.06:** 👥**Training Collaboration** is online, supporting project collaborators, sharing project links and QR codes; The workspace supports a list view, displaying project Tags;
*   **2025.07.29:** 🚀The sidebar supports **Experiment Filtering and Sorting**; 📊The table view is online **Column Control Panel**, which can easily hide and show columns; 🔐 **Multiple API Key** management is online to make your data more secure; swanlab sync improves the compatibility of log file integrity and adapts to scenarios such as training crashes; New charts-PR curve, ROC curve, confusion matrix online, [documentation](https://docs.swanlab.cn/api/py-pr_curve.html);
*   **2025.07.17:** 📊 More powerful **Line chart configuration**, supporting flexible configuration of line type, color, thickness, grid, legend location, etc.; 📹 Support the **swanlab.Video** data type, support recording and visualizing GIF format files; Global chart dashboard supports configuring the Y-axis and the maximum number of displayed experiments;
*   **2025.07.10:** 📚More powerful **text view**, support Markdown rendering and directional key switching, can be created by `swanlab.echarts.table` and `swanlab.Text`, [Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)
*   **2025.07.06:** 🚄 Support **resume breakpoint training**; New plugin **File Logger**; Integrated [ray](https://github.com/ray-project/ray) framework, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); Integrated [ROLL](https://github.com/volcengine/ROLL) framework, thanks to [@PanAndy](https://github.com/PanAndy), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)
*   **2025.06.27:** 📊 Support **small line chart partial zoom**; Support configuring **smoothness of a single line chart**; Significantly improved the interactive effect after zooming in on the image chart;
*   **2025.06.20:** 🤗Integrated [accelerate](https://github.com/huggingface/accelerate) framework, [PR](https://github.com/huggingface/accelerate/pull/3605), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html), enhancing the experience of experiment recording in distributed training;
*   **2025.06.18:** 🐜Integrated [AREAL](https://github.com/inclusionAI/AReaL) framework, thanks to [@xichengpro](https://github.com/xichengpro), [PR](https://github.com/inclusionAI/AReaL/pull/98), [documentation](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process); 🖱Support the mouse Hover to the experiment in the sidebar, highlighting the corresponding curve; Support cross-group comparison of line charts; Support setting experiment name cropping rules;
*   **2025.06.11:** 📊 Support **swanlab.echarts.table** data type, support pure text chart display; Support **Stretching interaction** for grouping to increase the number of charts displayed at the same time; The table view adds **index maximum/minimum** option;
<details><summary>Complete Update Log</summary>

*   2025.06.08: ♻️Support storing complete experiment log files locally, and upload local log files to the cloud/private deployment end through **swanlab sync**; Hardware monitoring supports **Hygon DCU**;
*   2025.06.01: 🏸Support **chart free drag and drop**; Support **ECharts custom charts**, adding 20+ chart types including bar charts, pie charts, and histograms; Hardware monitoring supports **Moore Threads GPU**; Integrated **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** framework;
*   2025.05.25: The logs support recording **standard error streams**, and the print information of frameworks such as PyTorch Lightning can be better recorded; Hardware monitoring supports **Moore Threads**; Added security protection for running command records, and API Keys will be automatically hidden;
*   2025.05.14: Support **Experiment Tags**; Support line chart **Log Scale**; Support **Group Dragging**; Greatly optimized the experience of uploading a large number of indicators; Add `swanlab.OpenApi` open interface;
*   2025.05.09: Support **Line chart creation**; The configuration chart function adds the **data source selection** function, supporting the display of different indicators in a single chart; Support generating **training project GitHub badge**;
*   2025.04.23: Support line chart **editing**, support free configuration of the X and Y axis data range and title style; Chart search supports **regular expressions**; Support hardware detection and monitoring of **Kunlunxin XPU**;
*   2025.04.11: Support line chart **local area selection**; Support global selection of step range for dashboard line charts; Support one-click hiding of all charts;
*   2025.04.08: Support **swanlab.Molecule** data type, support recording and visualizing biochemical molecular data; Support saving the sorting, filtering, and column order change status in the table view;
*   2025.04.07: We have completed the joint integration with [EvalScope](https://github.com/ModelScope/EvalScope), and now you can use SwanLab to **evaluate large model performance** in EvalScope;
*   2025.03.30: Support **swanlab.Settings** method, support more refined experiment behavior control; Support **Cambricon MLU** hardware monitoring; Support [Slack Notification](https://docs.swanlab.cn/plugin/notification-slack.html), [Discord Notification](https://docs.swanlab.cn/plugin/notification-discord.html);
*   2025.03.21: 🎉🤗HuggingFace Transformers has officially integrated SwanLab (>=4.50.0 version), [#36433](https://github.com/huggingface/transformers/pull/36433); Added **Object3D chart**, supporting the recording and visualization of three-dimensional point clouds, [Documentation](https://docs.swanlab.cn/api/py-object3d.html); Hardware monitoring supports the recording of GPU video memory (MB), disk utilization, and network up and down lines;
*   2025.03.12: 🎉🎉SwanLab **Private Deployment Edition** is now available!! [🔗Deployment Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab supports plugin extensions, such as [Email Notification](https://docs.swanlab.cn/plugin/notification-email.html), [Feishu Notification](https://docs.swanlab.cn/plugin/notification-lark.html)
*   2025.03.09: Support **Experiment Sidebar Widening**; Added external Git code button; Added **sync_mlflow** function, supporting synchronization of experimental tracking with the mlflow framework;
*   2025.03.06: We have completed the joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio), and now you can use SwanLab in DiffSynth Studio to **track and visualize Diffusion model text-to-image/video experiments**, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html);
*   2025.03.04: Added **MLFlow Conversion** function, supporting the conversion of MLFlow experiments to SwanLab experiments, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html);
*   2025.03.01: Added the **Move Experiments** function, now you can move the experiments to different projects of different organizations;
*   2025.02.24: We have completed the joint integration with [EasyR1](https://github.com/hiyouga/EasyR1), and now you can use SwanLab to **track and visualize multi-modal large model reinforcement learning experiments**, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
*   2025.02.18: We have completed the joint integration with [Swift](https://github.com/modelscope/ms-swift), and now you can use SwanLab in Swift's CLI/WebUI to **track and visualize large model fine-tuning experiments**, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html).
*   2025.02.16: Added the function of **Moving and Creating Groups for Charts**.
*   2025.02.09: We have completed the joint integration with [veRL](https://github.com/volcengine/verl), and now you can use SwanLab to **track and visualize large model reinforcement learning experiments** in veRL, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html).
*   2025.02.05: `swanlab.log` supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), adapting to Jax framework features; Supports `name` and `notes` parameters;
*   2025.01.22: Added the functions `sync_tensorboardX` and `sync_tensorboard_torch`, supporting the synchronization of experimental tracking with these two TensorBoard frameworks;
*   2025.01.17: Added the `sync_wandb` function, [Documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html), supporting synchronization with Weights & Biases experimental tracking; Significantly improved log rendering performance
*   2025.01.11: The cloud version significantly optimizes the performance of project tables and supports drag-and-drop, sorting, filtering and other interactions
*   2025.01.01: Added persistent smoothing of line charts, drag-and-drop resizing of line charts, and optimized chart browsing experience
*   2024.12.22: We have completed the joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), and now you can use SwanLab in LLaMA Factory to **track and visualize large model fine-tuning experiments**, [Usage Guide](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger).
*   2024.12.15: **Hardware monitoring (0.4.0)** function is online, supporting the recording and monitoring of system-level information of CPU, NPU (Ascend), GPU (Nvidia).
*   2024.12.06: Added integration for [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html) and [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html); Improved the limit on the length of a single line of log recording.
*   2024.11.26: The hardware section of the environment tab supports the recognition of **Huawei Ascend NPU** and **Kunpeng CPU**; The cloud vendor section supports the recognition of Qingyun **Basis Intelligence**.

</details>

<br>

## 📃 Online Demos

Explore SwanLab in action with these interactive demos:

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Track a simple ResNet50 model trained on the cat/dog dataset for image classification. | Track training hyperparameters and metrics using Yolov8 for object detection on the COCO128 dataset. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Track instruction fine-tuning of the Qwen2 large language model, completing simple instruction following. | Use a simple LSTM model trained on the Google stock dataset to predict future stock prices. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiments on audio classification from ResNet to ResNeXt | LoRA fine-tuning based on the Qwen2-VL multi-modal large model on the COCO2014 dataset. |

| [EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Multi-modal LLM RL training using the EasyR1 framework | GRPO training based on the Qwen2.5-0.5B model on the GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

If you want to experience the latest features, you can install from the source code.

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

1.  [Register a free account](https://swanlab.cn)

2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).

3.  Open your terminal and enter:

    ```bash
    swanlab login
    ```

    When prompted, enter your API Key and press Enter to complete the login.

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

That's it! Now, head over to [SwanLab](https://swanlab.cn) to view your first SwanLab experiment.

<br>

## 💻 Self-Hosting

The self-hosted community version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy the Self-Hosted Version using Docker

Refer to the documentation for details: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For quick installation in China:

```bash
./install.sh
```

To install from DockerHub images:

```bash
./install-dockerhub.sh
```

### 2. Point Your Experiments to Your Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, you can record experiments to the self-hosted service.

<br>

## 🔥 Real-World Examples

**Open-Source Projects with SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): Principle and practice tutorial for large language models from scratch ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm): 《Open Source Large Model Eating Guide》 tailored for Chinese users, based on a Linux environment for quickly fine-tuning (full parameters/Lora), deploying domestic and foreign open source large models (LLM)/multi-modal large models (MLLM) tutorial ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, expansion and reproduction ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
*   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): The vision head of SmolVLM2 is spliced and fine-tuned with the Qwen3-0.6B model. ![GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)

**Papers using SwanLab:**

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
*   [DQN Reinforcement Learning - Cartpole Inverted Pendulum](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pre-training](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multi-modal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multi-modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA and SwanLab Visualization Recording](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)

🌟If you have a tutorial you'd like to include, feel free to submit a PR!

<br>

## 🎮 Hardware Monitoring

SwanLab records and monitors the hardware information and resource usage during AI training. Here's the support table:

| Hardware        | Information Recording | Resource Monitoring | Script                                                                                                |
| --------------- | --------------------- | ------------------- | ----------------------------------------------------------------------------------------------------- |
| NVIDIA GPU      | ✅                    | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU      | ✅                    | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC       | ✅                    | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU   | ✅                    | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU   | ✅                    | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅                    | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU      | ✅                    | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU      | ✅                    | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU             | ✅                    | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)         |
| Memory          | ✅                    | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)     |
| Disk            | ✅                    | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)     |
| Network         | ✅                    | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)     |

If you want to record other hardware, welcome to submit an Issue and a PR!

<br>

## 🚗 Framework Integrations

Combine your favorite frameworks with SwanLab!  
Here is the list of frameworks we have integrated, and you are welcome to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to provide feedback on the frameworks you want to integrate.

**Base Frameworks**
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

**Evaluation Frameworks**
-   [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)

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

Extend SwanLab's functionality with plugins to enhance your experiment management experience!

-   [Customize Your Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email Notification](https://docs.swanlab.cn/plugin/notification-email.html)
-   [Feishu Notification](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notification](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [WorkWeChat Notification](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord Notification](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack Notification](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [File Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open APIs:
-   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚