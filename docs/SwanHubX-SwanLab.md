<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br/>

# SwanLab: Open-Source Deep Learning Experiment Tracking and Visualization

**SwanLab empowers AI researchers with a modern, open-source platform to track, visualize, and collaborate on deep learning experiments, supporting 30+ frameworks and offering cloud and self-hosted options.**

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![SwanLab Cloud](https://img.shields.io/badge/Product-SwanLab云端版-636a3f?labelColor=black&style=flat-square)](https://swanlab.cn/)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

<br/>

[中文 / English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features of SwanLab:

*   **Experiment Tracking & Visualization:**
    *   Track metrics, hyperparameters, and model outputs with minimal code integration.
    *   Visualize training progress with intuitive charts and dashboards.
    *   Support for various data types: Scalar metrics, images, audio, text, video, 3D point clouds, and custom Echarts.
    *   Comprehensive chart types: Line charts, media plots, 3D point cloud visualization, chemical molecule visualization, and custom Echarts support.
    *   Interactive text-based LLM output viewer with markdown rendering.
    *   Automatic logging of experiment metadata.
    *   Hardware monitoring for CPU, GPU, and more.
    *   Resume interrupted training sessions.
    *   Easily manage experiments with a dashboard, compare results, and see results in the cloud or self-hosted.
*   **Broad Framework Integration:** Seamlessly integrates with over 30 popular deep learning frameworks, including:
    *   PyTorch, Hugging Face Transformers, PyTorch Lightning, LLaMA Factory, and more.
*   **Hardware Monitoring:** Real-time monitoring and logging of hardware metrics, including:
    *   CPU, GPU (Nvidia, AMD, Intel, MetaX, Moore Threads), Ascend NPU, MLU, XPU, Memory, Disk, and Network usage.
*   **Flexible Experiment Management:**
    *   Centralized dashboards to manage multiple projects and experiments.
    *   Advanced data comparison and intuitive visualization of results, empowering iterative improvements.
    *   Enable collaboration by inviting your team to share and view experiments.
*   **Collaboration & Sharing:**
    *   Share experiments easily with persistent URLs.
    *   Collaborate on training runs with team members.
    *   Share findings and suggestions.
*   **Self-Hosting:**
    *   Run SwanLab locally or on your own servers for full control over your data.
*   **Extensible with Plugins:**
    *   Enhance functionality through plugins like: Email, Slack, CSV writer, and file log directory.
    *   OpenAPI available for custom integrations.

>   **Star the project** to stay updated on all releases and features! ⭐️

![star-us](readme_files/star-us.png)

<br/>

## Getting Started

### Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
cd SwanLab
pip install -e .
```

</details>

<details><summary>Offline Dashboard Installation</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### Login and API Key

1.  [Register for a free account](https://swanlab.cn/)
2.  Log in to your account and find your API key: User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and run:

```bash
swanlab login
```

Follow the prompts to enter your API key.

### Integrate SwanLab with Your Code

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

View your experiment results at [SwanLab](https://swanlab.cn).

<br/>

## Self-Hosted

SwanLab offers a self-hosted community version for offline use.

![swanlab-docker](./readme_files/swanlab-docker.png)

### Deploy with Docker

Refer to the documentation: [Self-Hosted Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

China Fast Install:

```bash
./install.sh
```

From DockerHub:

```bash
./install-dockerhub.sh
```

### Connect Experiments to Self-Hosted Instance

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Then all experiments will be logged into your self-hosted service.

<br/>

## Practical Examples

**Open-Source Projects Using SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): Large Language Model tutorial
*   [self-llm](https://github.com/datawhalechina/self-llm): Guide to open-source LLMs, for fine-tuning/deployment
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek model understanding and reproduction

**Research Papers Citing SwanLab:**

*   [Animation Needs Attention](https://arxiv.org/abs/2507.03916)

**Tutorials:**

*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [YOLO Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Tasks](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL multimodal fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO LLM Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B multimodal model training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied AI Tutorial](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)

<br/>

## Hardware Monitoring Support

SwanLab tracks hardware and resource usage during AI training. Supported hardware:

| Hardware       | Info Record | Resource Monitoring | Script                                               |
| -------------- | ----------- | ------------------- | ---------------------------------------------------- |
| Nvidia GPU     | ✅           | ✅                   | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)    |
| Ascend NPU     | ✅           | ✅                   | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)    |
| Apple SOC      | ✅           | ✅                   | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)      |
| Cambricon MLU  | ✅           | ✅                   | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU  | ✅           | ✅                   | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅           | ✅                   | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU | ✅           | ✅                   | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU  | ✅           | ✅                   | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py)  |
| CPU            | ✅           | ✅                   | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)          |
| Memory         | ✅           | ✅                   | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)       |
| Disk           | ✅           | ✅                   | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)          |
| Network | ✅           | ✅                   | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)         |

Contribute to add more hardware support by submitting an Issue or Pull Request!

<br/>

## Framework Integrations

Enhance your workflow with SwanLab by integrating it with your favorite frameworks!

**Base Frameworks**

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized / Fine-tuning Frameworks**

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

[More integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br/>

## Plugins & API

Extend SwanLab's functionality with plugins!

*   [Create Your Own Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Directory Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI:

*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br/>

## Comparison: TensorBoard vs. Weights & Biases vs. SwanLab

**Key Advantages of SwanLab:**

### TensorBoard vs SwanLab

*   **Cloud Support:** Sync training experiments to the cloud, making it easy to track progress remotely, manage historical projects, share experiment links, and collaborate. TensorBoard is an offline experiment tracking tool.
*   **Collaboration:**  SwanLab enables team-based machine learning. Easily manage training projects, share links, and communicate across teams. TensorBoard is mostly for individual use.
*   **Centralized Dashboard:**  Your results are in one dashboard, regardless of where your model trains—local machine, cloud, or cluster. TensorBoard requires time for managing and copying TFEvent files.
*   **Enhanced Tables:** SwanLab's tables are designed for large projects, allowing you to browse, search, and filter through results and compare thousands of model versions.

### Weights and Biases vs SwanLab

*   SwanLab offers a fully open-source, free, and self-hosted alternative, unlike Weights & Biases, which is a closed-source, cloud-based MLOps platform.

<br/>

## Community

### Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Documentation
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard web code
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosted deployment scripts

### Community and Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report issues and ask questions.
*   [Email Support](zeyi.lin@swanhub.co): Contact for specific issues.
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Discuss SwanLab, and share AI technical knowledge.

### SwanLab README Badges

Add SwanLab badges to your README:

[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![SwanLab Visualization](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More assets: [assets](https://github.com/SwanHubX/assets)

### Cite SwanLab

If SwanLab helped your research, please cite it using this format:

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

Read the [Contributing Guide](CONTRIBUTING.md) to get started.

We also greatly appreciate support from social media and conference presentations!

<br/>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br/>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

<br/>

## License

SwanLab is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)