<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## SwanLab: Open-Source AI Training Tracker and Visualizer

**Track, visualize, and collaborate on your AI experiments with SwanLab, the open-source, modern deep learning training tracker!**  Effortlessly integrate with 30+ popular frameworks and manage your machine learning projects with ease. Visit the [original repo](https://github.com/SwanHubX/SwanLab).

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

<div align="center">
  <img src="readme_files/swanlab-overview.png" alt="SwanLab Overview" width="80%">
</div>

<p align="center">
    <a href="https://github.com/SwanHubX/SwanLab/blob/main/README_EN.md">English</a> / <a href="https://github.com/SwanHubX/SwanLab/blob/main/README_JP.md">日本語</a> / <a href="https://github.com/SwanHubX/SwanLab/blob/main/README_RU.md">Русский</a>
</p>

<p align="center">
  👋 Join our <a href="https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html">WeChat Group</a>
</p>

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br>

## Key Features of SwanLab

*   **Experiment Tracking & Visualization:**  Visually track and analyze your model training metrics in real-time.

    *   Track scalar metrics, images, audio, text, videos, 3D point clouds, and custom Echarts charts.
    *   Visualize training progress with line charts, media views, and custom charts.
    *   LLM-specific visualization components for text content, with Markdown rendering support.

*   **Comprehensive Framework Integrations:** Seamlessly integrates with over **30** popular machine learning frameworks, including:
    *   PyTorch, HuggingFace Transformers, PyTorch Lightning, LLaMA Factory, MMDetection, Ultralytics, PaddleDetetion, LightGBM, XGBoost, Keras, Tensorboard, Weights&Biases and many more.

    <div align="center">
    <img src="readme_files/integrations.png" alt="Framework Integrations" width="80%">
    </div>

*   **Hardware Monitoring:** Monitor system-level hardware metrics during training for efficient resource management.

    *   Support for CPU, NPU (Ascend), GPU (Nvidia, and more), MLU (Cambricon), XLU (Kunlunxin), DCU (Hygon), MetaX GPU (MXGPU), Moore Threads GPU, and memory monitoring.

*   **Experiment Management & Comparison:**  Organize and compare your experiments efficiently.

    *   Centralized dashboard for managing projects and experiments.
    *   Compare results via online tables and charts to find insights.

*   **Collaboration & Sharing:**  Facilitate collaborative training and share results with ease.

    *   Real-time synchronization of experiments within a project.
    *   Share experiments with persistent URLs.

*   **Self-Hosting Support:** Utilize SwanLab in offline environments with the self-hosted community version.

*   **Plugin Extensibility:** Extend SwanLab's functionality with plugins like email notifications, file loggers, and more.

### Recent Updates

*   **(2025.08.19):** Enhanced chart rendering performance and low-intrusive loading animations for improved focus on experiment analysis. Integration of the excellent MLX-LM and SpecForge frameworks for broader training experiences.

*   **(2025.08.06):** Launched training collaboration features, supporting project collaborator invitations and project link/QR code sharing. Workspace enhancements include list view and project tag display.

*   **(2025.07.29):** Improved experiment filtering and sorting in the sidebar. Added column control panels to the table view for easier column hiding and display. Introduced multi-API Key management for enhanced data security. Enhanced swanlab sync for better log file integrity compatibility in training crash scenarios. New charts: PR curve, ROC curve, and confusion matrix.

*   **(2025.07.17):** More powerful line chart configuration, supporting flexible configuration of line styles, colors, thickness, grids, and legend positions, along with support for swanlab.Video data types.

*   **(2025.07.10):** Enhanced text view with Markdown rendering and arrow key navigation capabilities, created by `swanlab.echarts.table` and `swanlab.Text`.

*   **(2025.07.06):** Supported resume training, new file logger plugin, and integrated the Ray and ROLL frameworks.

*   **(2025.06.27):** Support local zoom for line charts, single line chart smoothing configuration, and significant improvements to image chart zoom interactions.

*   **(2025.06.20):** Integrated the accelerate framework, enhancing the experiment recording experience in distributed training.

*   **(2025.06.18):** Integrated the AREAL framework, with support for mouse hover highlighting of corresponding curves in the sidebar experiments, cross-group comparison of line charts, and experimental name cropping rules settings.

<details><summary>Complete Changelog</summary>

- 2025.06.11: Added swanlab.echarts.table data type for pure text chart display; group stretching interaction to increase simultaneous chart display numbers; added index max/min options to the table view.
- 2025.06.08: Support for complete experimental log file storage, allowing for local log file upload via swanlab sync; added support for Hygon DCU hardware monitoring.
- 2025.06.01: Chart free dragging supported; ECharts custom charts support, including 20+ chart types; support for MX GPU hardware monitoring.
- 2025.05.25: Standard error stream logging support, improved logging for PyTorch Lightning and other frameworks.
- 2025.05.14: Experiment Tag support, Log Scale support for line charts, group dragging support, and large-scale optimization of index uploads; added swanlab.OpenApi open interface.
- 2025.05.09: Added line chart creation support; chart configuration features added data source selection function, supporting a single chart display of different indicators; support training project GitHub badges.
- 2025.04.23: Support line chart editing, support the flexible configuration of the X, Y-axis data range and title style for the chart; Chart search supports regular expressions; Support hardware detection and monitoring for Kunlunxin XPU.
- 2025.04.11: Supports local area selection in line charts; supports global selection of step ranges for the dashboard line chart; support one-click hide of all charts.
- 2025.04.08: Support swanlab.Molecule data type, support recording and visualizing biochemical molecular data; support the saving of sorting, filtering, and column order changes in the table view.
- 2025.04.07: SwanLab has been jointly integrated with EvalScope, and now you can use SwanLab in EvalScope to evaluate large model performance.
- 2025.03.30: Supports swanlab.Settings method and more refined experimental behavior control; support Cambricon MLU hardware monitoring; support Slack notification, Discord notification.
- 2025.03.21: 🎉🤗HuggingFace Transformers has officially integrated SwanLab (>=4.50.0 version), [#36433](https://github.com/huggingface/transformers/pull/36433); added the Object3D chart, supporting the recording and visualization of three-dimensional point cloud, [Documentation](https://docs.swanlab.cn/api/py-object3d.html); hardware monitoring supports GPU memory (MB), disk utilization, and network upload and download recording.
- 2025.03.12: 🎉🎉SwanLab private deployment version has been released!![🔗Deployment Document](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab has supported plugin extension, such as email notification, Lark notification.
- 2025.03.09: Support the experimental sidebar widening; added the display Git code button; added the sync_mlflow function, supporting the synchronization of experimental tracking with the mlflow framework.
- 2025.03.06: We have completed the joint integration with DiffSynth Studio, and now you can use SwanLab in DiffSynth Studio to track and visualize Diffusion model text-to-image/video experiments, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html).
- 2025.03.04: Added MLFlow conversion function, supports the conversion of MLFlow experiments to SwanLab experiments, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html).
- 2025.03.01: Added the function of moving experiments, you can now move experiments to different projects of different organizations.
- 2025.02.24: We have completed the joint integration with EasyR1, and now you can use SwanLab in EasyR1 to track and visualize multimodal large model reinforcement learning experiments, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
- 2025.02.18: We have completed the joint integration with Swift, and now you can use SwanLab in Swift's CLI/WebUI to track and visualize large model fine-tuning experiments, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html).
- 2025.02.16: Added the function of chart moving grouping, creating grouping.
- 2025.02.09: We have completed the joint integration with veRL, and now you can use SwanLab in veRL to track and visualize large model reinforcement learning experiments, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html).
- 2025.02.05: `swanlab.log` supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), adapting to Jax framework characteristics; supports `name` and `notes` parameters.
- 2025.01.22: Added `sync_tensorboardX` and `sync_tensorboard_torch` functions, support the synchronization of experimental tracking with these two TensorBoard frameworks.
- 2025.01.17: Added `sync_wandb` function, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html), supporting synchronization with Weights & Biases experimental tracking; significantly improved log rendering performance
- 2025.01.11: The cloud version has greatly optimized the performance of the project table, and supports drag and drop, sorting, filtering and other interactions.
- 2025.01.01: Added line chart persistence smoothing, line chart drag and drop to change size, optimized the chart browsing experience
- 2024.12.22: We have completed the joint integration with LLaMA Factory, and now you can use SwanLab in LLaMA Factory to track and visualize large model fine-tuning experiments, [usage guide](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger).
- 2024.12.15: Hardware Monitoring (0.4.0) function is online, supporting system-level information recording and monitoring of CPU, NPU (Ascend), and GPU (Nvidia).
- 2024.12.06: Added integration for LightGBM and XGBoost; increased the limit on the length of a single line of log records.
- 2024.11.26: The hardware section of the environment tab supports the identification of Huawei Ascend NPU and Kunpeng CPU; the cloud vendor section supports the identification of Qingyun Cornerstone Intelligent Computing.

</details>

<br>

## Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# or

pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Expansion Installation</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Log in and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  In your terminal:

```bash
swanlab login
```

Enter your API Key when prompted.

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

Visit [SwanLab](https://swanlab.cn) to view your experiment.

<br>

## Practical Examples

*   **Tutorials**:
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
    *   [Qwen2-VL Multimodal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
    *   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
    *   [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
    *   [LeRobot Embodied Intelligence Guide](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
    *   [GLM-4.5-Air-LoRA and SwanLab Visualization](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
    *   [How to do RAG? SwanLab Document Assistant Solution is Open Sourced](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

*   **Open Source Projects & Papers**: Browse example projects using SwanLab:
    *   [happy-llm](https://github.com/datawhalechina/happy-llm): Comprehensive tutorial on the principles and practice of large language models.
    *   [self-llm](https://github.com/datawhalechina/self-llm): Tutorial on fine-tuning, deploying, and visualizing open-source LLMs/MLLMs.
    *   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series interpretation, expansion, and reproduction.
    *   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): Fine-tuning the visual head of SmolVLM2 with the Qwen3-0.6B model.

    *   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
    *   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
    *   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
    *   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

    **Contribute:**  Submit a PR with your tutorial to be included here!

<br>

## Hardware Monitoring

| Hardware            | Information Recording | Resource Monitoring | Script                                                                        |
| ------------------- | --------------------- | ------------------- | ----------------------------------------------------------------------------- |
| NVIDIA GPU          | ✅                    | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU          | ✅                    | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC           | ✅                    | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU       | ✅                    | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU       | ✅                    | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU   | ✅                    | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU           | ✅                    | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU           | ✅                    | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU                 | ✅                    | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)     |
| Memory              | ✅                    | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk                | ✅                    | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)   |
| Network             | ✅                    | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Add support for your hardware - submit an issue or PR!

<br>

## Framework Integrations

*   **Basic Frameworks**:
    *   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
    *   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
    *   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

*   **Specialized/Fine-tuning Frameworks**:
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

*   **Evaluation Frameworks**:
    *   [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)

*   **Computer Vision**:
    *   [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
    *   [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
    *   [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
    *   [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
    *   [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)

*   **Reinforcement Learning**:
    *   [Stable Baseline3](https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html)
    *   [veRL](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)
    *   [HuggingFace trl](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-trl.html)
    *   [EasyR1](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
    *   [AReaL](https://docs.swanlab.cn/guide_cloud/integration/integration-areal.html)
    *   [ROLL](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

*   **Other Frameworks**:
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

## Plugins and API

*   [Customize your Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)

*   Notification Plugins:
    *   [Email](https://docs.swanlab.cn/plugin/notification-email.html)
    *   [Feishu](https://docs.swanlab.cn/plugin/notification-lark.html)
    *   [DingTalk](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
    *   [WeChat Work](https://docs.swanlab.cn/plugin/notification-wxwork.html)
    *   [Discord](https://docs.swanlab.cn/plugin/notification-discord.html)
    *   [Slack](https://docs.swanlab.cn/plugin/notification-slack.html)

*   Writer Plugins:
    *   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
    *   [File Log Directory Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

*   OpenAPI:
    *   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparisons with Similar Tools

*   **TensorBoard vs. SwanLab**: SwanLab offers cloud support, team collaboration, a persistent dashboard, and enhanced tables. TensorBoard is primarily an offline tool.
*   **Weights & Biases vs. SwanLab**: SwanLab is open-source, free, and supports self-hosting, providing similar functionality to Weights & Biases's paid platform.

<br>

## Community

*   **Repositories**:
    *   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Documentation repository
    *   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard web code
    *   [self-hosted](https://github.com/swanhubx/self-hosted): Private deployment script repository

*   **Community and Support**:
    *   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): For bugs and issues
    *   [Email Support](zeyi.lin@swanhub.co): For general feedback
    *   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): For questions and discussions

*   **SwanLab README Badges**:

    Add SwanLab badges to your README:

    [![][tracking-swanlab-shield]][tracking-swanlab-shield-link]
    [![][visualize-swanlab-shield]][visualize-swanlab-shield-link]

    ```markdown
    [![SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
    [![SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
    ```

    More design assets are available here: [assets](https://github.com/SwanHubX/assets)

*   **Citing SwanLab in your research:**

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

*   **Contributing to SwanLab:**

    Review the [Contribution Guide](CONTRIBUTING.md). We appreciate contributions via social media, events, and conferences!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

<br>

## License

This project is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)

<!-- link -->

[release-shield]: https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square
[release-link]: https://github.com/swanhubx/swanlab/releases

[license-shield]: https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square
[license-shield-link]: