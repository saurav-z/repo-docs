<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br>

## 🚀 SwanLab: Open-Source AI Experiment Tracking & Visualization

**SwanLab is an open-source tool designed to supercharge your deep learning workflow by tracking, visualizing, and collaborating on your AI experiments.** Easily integrate with 30+ popular frameworks, access powerful visualization tools, and collaborate with your team, all in a user-friendly interface. Explore the [SwanLab GitHub Repository](https://github.com/SwanHubX/SwanLab) to get started!

*   [🔥 SwanLab Online](https://swanlab.cn)
*   [📃 Documentation](https://docs.swanlab.cn)
*   [Report Issues](https://github.com/swanhubx/swanlab/issues)
*   [Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc)
*   [Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html)
*   [Baseline Community](https://swanlab.cn/benchmarks)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

<div align="center">
<img src="readme_files/swanlab-overview.png">
</div>
<br>
<div align="center">
中文 / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)
</div>

<br>

## Key Features

*   **📊 Experiment Tracking & Visualization:**
    *   Effortlessly track metrics, hyperparameters, and system resources.
    *   Visualize training progress with interactive charts and graphs.
    *   Support for various data types: scalar metrics, images, audio, text, videos, 3D point clouds, and custom Echarts.
    *   Rich chart types including line plots, media views, 3D point clouds, and custom Echarts.
    *   Text-based LLM output visualization with Markdown rendering.

<div align="center">
<img src="readme_files/molecule.gif">
</div>

*   **⚡️ Extensive Framework Integrations:** Seamlessly integrate with 30+ popular machine learning frameworks, including PyTorch, Hugging Face Transformers, PyTorch Lightning, LLama Factory, and more.

<div align="center">
<img src="readme_files/integrations.png" />
</div>

*   **💻 Hardware Monitoring:** Real-time monitoring of CPU, GPU, NPU (Ascend), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (MX), Moore Threads GPU, Memory, and more.
*   **📦 Experiment Management:** Intuitive dashboard for managing projects and experiments.
*   **🆚 Comparison & Collaboration:** Compare experiments, analyze results, and collaborate with your team.
*   **✉️ Share Results:** Share experiment results via persistent URLs.
*   **💻 Self-Hosting:** Run SwanLab locally or on your own server.
*   **🔌 Plugin Ecosystem:** Extend functionality with plugins for notifications, data logging, and more.
*   **✅ Supports resume breakpoint and automatic log recording.**

<br>

## Getting Started

### 1. Installation

```bash
pip install swanlab
```

### 2. Login & Get API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Get your API key from [API Key](https://swanlab.cn/settings).
3.  Open terminal and login:

```bash
swanlab login
```

Enter your API key when prompted.

### 3. Integrate SwanLab into your Code

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

Now, visit [SwanLab](https://swanlab.cn) to view your experiment.

<br>

## 💻 Self-Hosting

Self-hosted community version supports offline viewing the SwanLab dashboard.

<div align="center">
<img src="./readme_files/swanlab-docker.png">
</div>

### 1. Deploying the Self-Hosted Version with Docker

See the documentation for details: [Docs](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Fast installation in China:

```bash
./install.sh
```

Install by pulling images from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Specifying Experiments to Self-Hosted Services

Login to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, you can record experiments to the self-hosted service.

<br>

## 🔥 Real-World Examples

See our tutorials and examples:

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
*   [Qwen2-VL Multimodal Large Model Fine-tuning Practice](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA and SwanLab Visualization Recording](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)

<br>

## 🚗 Framework Integrations

Use SwanLab in conjunction with your favorite frameworks!

**Basic Frameworks**
-   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
-   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
-   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Special/Fine-tuning Frameworks**
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

Enhance your experiment management experience through plugins!

-   [Customize your plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email notification](https://docs.swanlab.cn/plugin/notification-email.html)
-   [Feishu Notification](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notification](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [WeChat Work Notification](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord notification](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack notification](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV recorder](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [File recorder](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open APIs:
-   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparison with Similar Tools

### Tensorboard vs SwanLab

*   **☁️ Online Support:** SwanLab is an online experiment tracking and visualization tool, allowing you to view training progress remotely, manage historical projects, share experiment links, send real-time notifications, and view experiments on multiple devices. Tensorboard is a local experiment tracking tool.
*   **👥 Collaboration:** SwanLab facilitates collaborative machine learning, allowing you to easily manage projects, share experiment links, and exchange ideas with teams. Tensorboard is primarily designed for individual use and lacks collaboration features.
*   **💻 Persistent, Centralized Dashboard:** Track results in a centralized dashboard, regardless of your training location (local, cluster, or cloud). Using TensorBoard requires managing TFEvent files manually from different machines.
*   **💪 More Powerful Table:** SwanLab's table views make it easier to find the best-performing models by viewing, searching, and filtering results from different experiments. TensorBoard is not designed for large projects.

### Weights and Biases vs SwanLab

*   Weights and Biases is a proprietary and cloud-based MLOps platform.
*   SwanLab supports both cloud-based and open-source, free, self-hosted versions.

<br>

## 👥 Community

### Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Web code for the lightweight offline dashboard, opened by `swanlab watch`
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting deployment scripts

### Community & Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report bugs or ask questions.
*   [Email](zeyi.lin@swanhub.co): For feedback on using SwanLab.
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discuss issues and share your projects.

### SwanLab README Badges

Add the SwanLab badge to your README:

```
[![SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in your Research

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

### Contributing

Read our [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

<br>

## 📃 License

This project is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)