<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
    <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
  </picture>

  **Supercharge your machine learning workflow with SwanLab, the open-source, modern tool for tracking, visualizing, and collaborating on your AI experiments.**

  <a href="https://swanlab.cn">🔥SwanLab Online</a> | <a href="https://docs.swanlab.cn">📃 Documentation</a> | <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> | <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> | <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> | <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks</a>

  [![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
  [![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
  [![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
  [![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
  [![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
  [![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
  [![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
  [![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
  [![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
  [![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

  ![](readme_files/swanlab-overview.png)

  [中文](README.md) / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

  👋 Join our <a href="https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html">WeChat Group</a>

  <a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</div>

<br/>

## Key Features

*   **Experiment Tracking & Visualization:**  Effortlessly track metrics, hyperparameters, and model outputs with an intuitive UI, enabling data-driven insights.
    *   📈 Visualize training metrics, including scalars, images, audio, text, video, 3D point clouds, and custom ECharts graphs.
    *   🗄️ Supports metadata logging for hyperparameters, metrics, and tabular analysis.
    *   🎨 Customizable charts for comprehensive analysis and easy comparison.
    *   🚀 Automatic logging of logs, hardware environment, Git repository, Python environment, and project run directories.
    *   🔄 Supports logging of breakpoints for continued metrics logging.
*   **Flexible Framework Integrations:** Seamless integration with over 30 popular machine learning frameworks.
    *   ⚙️ Includes PyTorch, TensorFlow/Keras, Hugging Face Transformers, PyTorch Lightning, and more.
    *   🧩 Easily extendable through a plugin-based architecture.
*   **Hardware Monitoring:**  Gain in-depth insights into your system's resource usage.
    *   💻 Real-time monitoring of CPU, GPU (Nvidia, Ascend, and others), memory, disk, and network.
*   **Experiment Management & Collaboration:** Streamline your workflow with powerful tools for managing and collaborating on experiments.
    *   🏢 Centralized dashboards for efficient project and experiment organization.
    *   🆚 Compare experiments side-by-side to discover key differences and insights.
    *   🧑‍🤝‍🧑 Enable collaborative training with project sharing, online viewing, and feedback.
    *   🔗 Share results easily with persistent URLs.
*   **Self-Hosted & Cloud-Ready:** Use SwanLab in the cloud or on your local machine.
    *   ☁️ Cloud version for remote monitoring and project organization.
    *   💻 Self-hosted version for offline use.
*   **Extensible through Plugins:** Enhance the functionality to suit your specific needs.
    *   ✉️ Notification plugins for instant updates.
    *   🔌 Custom plugins for extended functionality.
*   **Rich Data Types:** Support for various data types to visualize diverse model outputs.
    *   🖼️ Images, audio, text, video, 3D point clouds, biochemical molecules.

>  **Star the project** to stay updated! ⭐

<br>

## Getting Started

### 1. Installation

```bash
pip install swanlab
```

### 2. Login & Get API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Go to your settings > [API Key](https://swanlab.cn/settings) to get your API key.
3.  Open your terminal and type:

```bash
swanlab login
```

Enter your API key when prompted.

### 3. Integrate SwanLab into your code

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

<br>

## Examples & Tutorials

Access a variety of tutorials and example projects to start your machine learning journey with SwanLab.

*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [ResNet Cat and Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [YOLO Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Image Generation Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multimodal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence Guide](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA and SwanLab Visualization Recording](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
*   [How to do RAG? SwanLab Documentation Assistant](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

<br>

## Hardware Monitoring

SwanLab tracks hardware information and resource usage during AI training. The following hardware is supported:

*   Nvidia GPU
*   Ascend NPU
*   Apple SOC
*   Cambricon MLU
*   Kunlunxin XPU
*   Moore Threads GPU
*   Metax GPU
*   Hygon DCU
*   CPU
*   Memory
*   Disk
*   Network

## Framework Integrations

Integrate SwanLab with your favorite frameworks! We offer integrations with a wide range of tools.
*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)
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
*   [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)
*   [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
*   [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
*   [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
*   [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
*   [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)
*   [Stable Baseline3](https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html)
*   [veRL](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)
*   [HuggingFace trl](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-trl.html)
*   [EasyR1](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
*   [AReaL](https://docs.swanlab.cn/guide_cloud/integration/integration-areal.html)
*   [ROLL](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)
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

## Plugins and API

Extend SwanLab’s functionality with plugins to improve your experiment management experience!

*   [Custom Plugin Guide](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

## Comparison

### Tensorboard vs SwanLab

*   **Cloud-Ready:** SwanLab allows you to seamlessly sync your training experiments online, allowing for remote monitoring, project organization, experiment sharing, and more. Tensorboard is a local tool.
*   **Collaboration:** SwanLab simplifies multi-person, cross-team machine learning collaboration. Tensorboard is mainly designed for individual use.
*   **Centralized Dashboard:** Results are recorded in a centralized dashboard whether you train locally, on a cluster, or in the cloud. Tensorboard requires you to copy and manage TFEvent files from different machines.
*   **Enhanced Tables:** SwanLab offers powerful tables for browsing, searching, and filtering across experiments, making it easier to find the best models. TensorBoard doesn't scale well for large projects.

### Weights & Biases vs SwanLab

*   Weights & Biases is a closed-source MLOps platform that requires an internet connection.
*   SwanLab is not only for online use, but it is also open-source, free, and has a self-hosted version.

## Community

### Repositories
- [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs)：Official documentation repository
- [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard)：Offline dashboard repository.
- [self-hosted](https://github.com/swanhubx/self-hosted)：Self-hosting scripts repository.

### Support
- [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report issues or ask questions.
- [Email Support](zeyi.lin@swanhub.co): Get help with SwanLab.
- <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Get help with SwanLab.

### SwanLab README Badges

Add a SwanLab badge to your README:

[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in your paper

If you found SwanLab to be helpful for your research, please consider citing it:

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

### Contribute to SwanLab

Want to contribute to SwanLab? Please take a look at the [Contribution Guide](CONTRIBUTING.md) first.

Your support by sharing on social media, at events, and in meetings would be greatly appreciated!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## License

This repository is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)