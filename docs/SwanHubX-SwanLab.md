<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>
</div>
<br>

## SwanLab: Unleash the Power of Visualized AI Training 🚀

**SwanLab is an open-source, user-friendly tool for tracking, visualizing, and collaborating on your machine learning experiments, offering seamless integration with 30+ popular frameworks.** ([Back to Original Repo](https://github.com/SwanHubX/SwanLab))

<br/>

**Key Features:**

*   **📊 Experiment Tracking & Visualization:** Track key metrics, visualize training progress with interactive charts, and gain insights into your model's performance.
*   **📦 Comprehensive Framework Integration:** Seamlessly integrates with a wide range of frameworks, including PyTorch, Hugging Face Transformers, and more.
*   **💻 Hardware Monitoring:** Monitor CPU, GPU (Nvidia, Ascend, etc.), and other hardware metrics in real-time.
*   **✅ Offline & Cloud Support:** Use SwanLab locally or on the cloud for flexible experiment management.
*   **🤝 Collaboration & Sharing:** Share experiment results with colleagues and collaborate on projects for enhanced productivity.
*   **🌐 Rich Metadata Support**: Track scalar metrics, images, audio, text, video, 3D point clouds, biochemical molecules, and custom ECharts charts.
*   **🔌 Plugin Extensibility:** Extend SwanLab's functionality with plugins for notifications, data logging, and more.
*   **🚀 Multi-Platform Support**:  Built in Support for a large amount of hardware including, CPU, GPU (Nvidia, Moore Threads, Metax), Ascend, MLU and more.

<br/>

## Key Updates

*   **2025.08.19**: Enhanced chart rendering, low-intrusive loading animations, and integration of MLX-LM and SpecForge.
*   **2025.08.06**:  Training Collaboration features, including project collaborator invites, and workspace list view.
*   **2025.07.29**:  Experiment filtering, sorting in sidebar, and column control panel for table view. Multi-API key management.
*   **2025.07.17**:  Enhanced line chart configuration, GIF video support.
*   **2025.07.10**:  Enhanced text view with Markdown rendering and arrow key navigation, created with `swanlab.echarts.table` and `swanlab.Text`.
*   **2025.07.06**:  Resume support; new file logger plugin; integration with the Ray and ROLL frameworks.
*   **2025.06.27**:  Local zoom-in for line charts; smoothing option for individual line charts; improved image chart zoom interactions.
*   **2025.06.20**:  Integration with the accelerate framework; Enhancements to experiment tracking in distributed training.
*   **2025.06.18**:  Integration with the AREAL framework; Mouse hover for experiment highlighting in the sidebar; cross-group line chart comparison.

<details><summary>Complete Changelog</summary>

*   **2025.06.11**:  Support for `swanlab.echarts.table` data type, group stretching interaction, and column options.
*   **2025.06.08**:  Local experiment logs with `swanlab sync` upload to cloud/private deployment. Hardware monitoring support for Hygon DCU.
*   **2025.06.01**:  Chart dragging, ECharts customization, and support for more chart types. Hardware monitoring for Muxi GPU. PaddleNLP framework integration.
*   **2025.05.25**: Standard error stream logging; Hardware monitoring for Moore Threads. API key hiding.
*   **2025.05.14**: Experiment tags, Log Scale for line charts, group dragging.  `swanlab.OpenApi` open interface.
*   **2025.05.09**: Line chart creation, chart configuration, and training project badge generation.
*   **2025.04.23**: Line chart editing, hardware detection, and monitoring.
*   **2025.04.11**: Line chart region selection, global step range selection, and one-click chart hiding.
*   **2025.04.08**: `swanlab.Molecule` data type and table view state saving.
*   **2025.04.07**:  Joint integration with EvalScope to evaluate LLM performance.
*   **2025.03.30**:  `swanlab.Settings` method, MLU hardware monitoring, Slack and Discord notifications.
*   **2025.03.21**:  🎉🤗 HuggingFace Transformers integration; Object3D charts; GPU/disk/network monitoring.
*   **2025.03.12**:  🎉🎉 SwanLab private deployment release; plugin support.
*   **2025.03.09**:  Experiment sidebar expansion, Git code button, and MLflow sync.
*   **2025.03.06**:  Joint integration with DiffSynth Studio for Diffusion model experiments.
*   **2025.03.04**:  MLFlow conversion.
*   **2025.03.01**:  Move experiments.
*   **2025.02.24**:  Joint integration with EasyR1 for multimodal LLM RL experiments.
*   **2025.02.18**:  Joint integration with Swift for large model fine-tuning.
*   **2025.02.16**:  Chart group moving/creating.
*   **2025.02.09**:  Joint integration with veRL for large model RL experiments.
*   **2025.02.05**: `swanlab.log` for nested dictionaries. Support for `name` and `notes` parameters.
*   **2025.01.22**: `sync_tensorboardX` and `sync_tensorboard_torch` functions for syncing experiments.
*   **2025.01.17**: `sync_wandb` for Weights & Biases syncing and improved log rendering.
*   **2025.01.11**: Improved project table performance and drag/sort/filter functions.
*   **2025.01.01**: Line chart smoothing and drag-resize.
*   **2024.12.22**:  Integration with LLaMA Factory for large model fine-tuning.
*   **2024.12.15**:  Hardware monitoring (0.4.0) for CPU, NPU (Ascend), GPU (Nvidia).
*   **2024.12.06**:  Integration with LightGBM and XGBoost; increased log line limit.
*   **2024.11.26**:  Hardware recognition of Huawei Ascend NPU and Kunpeng CPU, along with support for QingCloud Jishi Zhisuan.
</details>

<br>

## 📃 Online Demos

Explore live demos showcasing SwanLab's capabilities:

| [ResNet50 Cat/Dog Classification][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Tracks a simple ResNet50 model on a cat/dog dataset. | Tracks YOLOv8 on the COCO128 dataset. |

| [Qwen2 Instruction Fine-tuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Tracks instruction fine-tuning for the Qwen2 language model. | Predicts future Google stock prices using an LSTM model. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Fine-tuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Tracks the process of ResNet to ResNeXt model experiments on audio classification tasks | Based on the Qwen2-VL multi-modal large model, LoRA fine-tuning is performed on the COCO2014 dataset. |

| [EasyR1 Multi-Modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| Train multi-modal LLM with the EasyR1 framework | Train GRPO based on the Qwen2.5-0.5B model on the GSM8k dataset |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Getting Started Quickly

### 1. Install

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
cd SwanLab
pip install -e . # or pip install -e .[dashboard] for offline dashboard
```

</details>

### 2. Login and Get Your API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Login and get your API Key from User Settings > [API Key](https://swanlab.cn/settings)
3.  Open the terminal and run:

```bash
swanlab login
```

Enter your API key when prompted.

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

<br>

## 💻 Self-Hosting

Self-host the community version for offline dashboard viewing.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy with Docker

Refer to the documentation for details: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Fast Installation for China:

```bash
./install.sh
```

Install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Direct Experiments to Your Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Log in and start recording to the self-hosted service.

<br>

## 🔥 Practical Cases and Examples

**Open Source Projects Utilizing SwanLab:**

-   [happy-llm](https://github.com/datawhalechina/happy-llm): Tutorial on large language model theory and practice
-   [self-llm](https://github.com/datawhalechina/self-llm): Large model fine-tuning and deployment tutorial
-   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series model overview and reproduction
-   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): Fine-tuning SmolVLM2's vision head with Qwen3-0.6B model

**Relevant Publications:**

-   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
-   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
-   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
-   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorials:**

-   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
-   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
-   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
-   [Resnet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
-   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
-   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
-   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
-   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
-   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
-   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
-   [Stable Diffusion Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
-   [LLM Pre-training](https://docs.swanlab.cn/examples/pretrain_llm.html)
-   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
-   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
-   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
-   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
-   [Qwen2-VL Multi-Modal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
-   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
-   [Qwen3-SmVL-0.6B Multi-Modal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
-   [LeRobot Embodied Intelligence](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
-   [GLM-4.5-Air-LoRA and SwanLab Visualization](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)

🌟 Submit a pull request if you want to add your tutorials!

<br>

## 🎮 Hardware Monitoring

SwanLab monitors and records **hardware information** and resource usage.

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
| Memory  | ✅        | ✅        | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk    | ✅        | ✅        | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Report an issue with your desired hardware for consideration.

<br>

## 🚗 Framework Integrations

Integrate SwanLab with your favorite frameworks!  
Check out the list of supported frameworks and file a [Issue](https://github.com/swanhubx/swanlab/issues) to request one.

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

**Other Frameworks**
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

Enhance your SwanLab experience with plugins!

-   [Custom Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
-   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV Logger](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [File Logger](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:
-   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparison with Familiar Tools

### Tensorboard vs SwanLab

-   **☁️ Online Support**: SwanLab enables convenient cloud synchronization, management of historical projects, sharing experiment links, real-time notifications, and multi-device viewing. Tensorboard is a local experiment tracking tool.

-   **👥 Multi-User Collaboration**: SwanLab simplifies the management of multi-person training projects, sharing experiment links, and cross-team discussions. Tensorboard is designed for individuals.

-   **💻 Persistent, Centralized Dashboard**: Regardless of your training environment, your results are recorded in a centralized dashboard. TensorBoard requires managing TFEvent files.

-   **💪 Enhanced Table**: SwanLab's table enables you to search, filter, and compare results from different experiments. TensorBoard is not suited for large projects.

### Weights and Biases vs SwanLab

-   Weights and Biases is a proprietary MLOps platform that needs an internet connection.

-   SwanLab supports both network and the open-source, free, self-hosted version.

<br>

## 👥 Community

### Additional Resources

-   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official Documentation
-   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline Dashboard Code
-   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting deployment scripts

### Community and Support

-   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report issues and ask questions.
-   [Email Support](zeyi.lin@swanhub.co): Contact the team.
-   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Engage with other users and discuss SwanLab.

### SwanLab README Badges

Add a SwanLab badge to your README:

[![][tracking-swanlab-shield]][tracking-swanlab-shield-link] [![][visualize-swanlab-shield]][visualize-swanlab-shield-link]

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

Find more design assets in [assets](https://github.com/SwanHubX/assets).

### Citing SwanLab in Publications

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

Read the [contribution guide](CONTRIBUTING.md) before contributing.

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## 📃 License

This project is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)

<!-- link -->

[release-shield]: https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square
[release-link]: https://github.com/swanhubx/swanlab/releases

[license-shield]: https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square
[license-shield-link]: https://github.com/SwanHubX/SwanLab/blob/main/LICENSE

[last-commit-shield]: https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square
[last-commit-shield-link]: https://github.com/swanhubx/swanlab/commits/main

[pypi-version-shield]: https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square
[pypi-version-shield-link]: https://pypi.org/project/swanlab/

[pypi-downloads-shield]: https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square
[pypi-downloads-shield-link]: https://pepy.tech/project/swanlab

[swanlab-cloud-shield]: https://img.shields.io/badge/Product-SwanLab云端版-636a3f?labelColor=black&style=flat-square
[swanlab-cloud-shield-link]: https://swanlab.cn/

[wechat-shield]: https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square
[wechat-shield-link]: https://docs.swanlab.cn/guide_cloud/community/online-support.html

[colab-shield]: https://colab.research.google.com/assets/colab-badge.svg
[colab-shield-link]: https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing

[github-stars-shield]: https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47
[github-stars-link]: https://github.com/swanhubx/swanlab

[github-issues-shield]: https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb
[github-issues-shield-link]: https://github.com/swanhubx/swanlab/issues

[github-contributors-shield]: https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square
[github-contributors-link]: https://github.com/swanhubx/swanlab/graphs/contributors

[demo-cats-dogs]: https://swanlab.cn/@ZeyiLin/Cats_Dogs_Classification/runs/jzo93k112f15pmx14vtxf/chart
[demo-cats-dogs-image]: readme_files/example-catsdogs.png

[demo-yolo]: https://swanlab.cn/@ZeyiLin/ultratest/runs/yux7vclmsmmsar9ear7u5/chart
[demo-yolo-image]: readme_files/example-yolo.png

[demo-qwen2-sft]: https://swanlab.cn/@ZeyiLin/Qwen2-fintune/runs/cfg5f8dzkp6vouxzaxlx6/chart
[demo-qwen2-sft-image]: readme_files/example-qwen2.png

[demo-google-stock]:https://swanlab.cn/@ZeyiLin/Google-Stock-Prediction/charts
[demo-google-stock-image]: readme_files/example-lstm.png

[demo-audio-classification]:https://swanlab.cn/@ZeyiLin/PyTorch_Audio_Classification/charts
[demo-audio-classification-image]: readme_files/example-audio-classification.png

[demo-qwen2-vl]:https://swanlab.cn/@ZeyiLin/Qwen2-VL-finetune/runs/pkgest5xhdn3ukpdy6kv5/chart
[demo-qwen2-vl-image]: readme_files/example-qwen2-vl.jpg

[demo-easyr1-rl]:https://swanlab.cn/@Kedreamix/easy_r1/runs/wzezd8q36bb6dlza6wtpc/chart
[demo-easyr1-rl-image]: readme_files/example-easyr1-rl.png

[demo-qwen2-grpo]:https://swanlab.cn/@kmno4/Qwen-R