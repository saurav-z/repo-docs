<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br/>

## SwanLab: Open-Source Deep Learning Experiment Tracking & Visualization

**SwanLab** is an open-source tool for tracking and visualizing deep learning experiments, offering a comprehensive platform for monitoring, analyzing, and collaborating on your AI projects. 

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)

[🔥SwanLab Online Version](https://swanlab.cn) | [📃 Documentation](https://docs.swanlab.cn) | [🐞 Report Issues](https://github.com/swanhubx/swanlab/issues) | [💬 Suggest Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) | [🚀 Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) |  <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> [🤝 Community](https://swanlab.cn/benchmarks)

<br/>

<img src="readme_files/swanlab-overview.png" alt="SwanLab Overview" width="100%">

[中文 / English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking:** Effortlessly log metrics, hyperparameters, and rich media to monitor your model's performance.
*   **Visualization:** Intuitive UI for visualizing training progress, identifying trends, and analyzing results.
*   **Framework Integrations:**  Seamlessly integrates with 30+ popular deep learning frameworks (PyTorch, TensorFlow, etc.).
*   **Hardware Monitoring:** Real-time monitoring of CPU, GPU (Nvidia, Ascend, etc.), and other hardware metrics.
*   **Collaboration & Sharing:** Share experiments, compare results, and collaborate with your team.
*   **Self-Hosting:** Supports both cloud and local/offline use, providing flexibility for all environments.
*   **Customization & Extensibility:**  Extensible through plugins for expanded functionality.
*   **Comprehensive Metadata Logging**: Auto-logging of logging, Git repository, Python environment, dependencies, and project runtime directories.

<br/>

## What's New
*   **2025.07.17:** 📊 Enhanced **line chart configuration**, supporting flexible customization of line types, colors, thicknesses, grids, and legend positions; 📹 Support for **swanlab.Video** data type, enabling the recording and visualization of GIF files; Global chart dashboard supports configuring the Y-axis and maximum display experiments;
*   **2025.07.10:** 📚 More powerful **text view**, supporting Markdown rendering and arrow key switching, which can be created by `swanlab.echarts.table` and `swanlab.Text`, [Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)
*   **2025.07.06:** 🚄 Support for **resume breakpoint training**; New plugin **file logger**; Integrated [ray](https://github.com/ray-project/ray) framework, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); Integrated [ROLL](https://github.com/volcengine/ROLL) framework, thanks [@PanAndy](https://github.com/PanAndy), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

*   [See more recent updates](https://github.com/SwanHubX/SwanLab/blob/main/README.md#-%E6%9C%80%E8%BF%91%E6%9B%B4%E6%96%B0)

<br/>

## Getting Started

### 1. Installation

```bash
pip install swanlab
```
  
<details><summary>Source Code Installation</summary>

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>
  
<details><summary>Offline Dashboard Installation</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Log in and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and run:

```bash
swanlab login
```

Enter your API key when prompted.

### 3. Integrate with Your Code

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

<br/>

## Live Demos

Explore SwanLab's capabilities with these interactive demos:

| [ResNet50 Cats vs. Dogs][demo-cats-dogs] | [Yolov8-COCO128 Object Detection][demo-yolo] |
| :--------: | :--------: |
| [![][demo-cats-dogs-image]][demo-cats-dogs] | [![][demo-yolo-image]][demo-yolo] |
| Track a simple ResNet50 model on the Cats & Dogs dataset. | Track YOLOv8 training on the COCO128 dataset. |

| [Qwen2 Instruction Finetuning][demo-qwen2-sft] | [LSTM Google Stock Prediction][demo-google-stock] |
| :--------: | :--------: |
| [![][demo-qwen2-sft-image]][demo-qwen2-sft] | [![][demo-google-stock-image]][demo-google-stock] |
| Track instruction finetuning of the Qwen2 large language model | Track training using a simple LSTM model on the Google Stock dataset. |

| [ResNeXt101 Audio Classification][demo-audio-classification] | [Qwen2-VL COCO Dataset Finetuning][demo-qwen2-vl] |
| :--------: | :--------: |
| [![][demo-audio-classification-image]][demo-audio-classification] | [![][demo-qwen2-vl-image]][demo-qwen2-vl] |
| Progressive experiments from ResNet to ResNeXt on audio classification tasks | Lora finetuning on COCO2014 dataset based on the Qwen2-VL multimodal large model. |

| [EasyR1 Multi-modal LLM RL Training][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO Training][demo-qwen2-grpo] |
| :--------: | :--------: |
| [![][demo-easyr1-rl-image]][demo-easyr1-rl] | [![][demo-qwen2-grpo-image]][demo-qwen2-grpo] |
| RL training on multi-modal LLM with EasyR1 framework | GRPO training on GSM8k dataset based on the Qwen2.5-0.5B model |

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br/>

## Self-Hosting

Self-hosted version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy with Docker

See the full guide: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Install quickly in China:

```bash
./install.sh
```

Or install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point Experiments to Self-Hosted

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Once logged in, your experiments will be recorded to the self-hosted service.

<br/>

## Real-World Use Cases

**Featured Open Source Projects Using SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): A tutorial on the principles and practice of large language models. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm): A tutorial focused on fine-tuning and deploying large language models. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series of work interpretation, extension, and reproduction. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)

**Tutorials & Example Projects:**

*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cats & Dogs Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Finetuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Finetuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Finetuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multimodal Model Finetuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO LLM Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence Tutorial](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)

<br/>

## Hardware Monitoring

SwanLab monitors hardware information and resource usage during AI training:

| Hardware | Info Recording | Resource Monitoring | Script |
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

Contribute by creating an issue and PR for any additional hardware logging needs!

<br/>

## Framework Integrations

Integrate your favorite framework with SwanLab!

Below is a list of frameworks currently supported:
  
**Base Frameworks**
*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Finetuning Frameworks**
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

<br/>

## Plugins and API

Enhance your experiment management with SwanLab's flexible plugin system!

*   [Custom Plugin Development](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeCom Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Directory Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br/>

## Comparisons

SwanLab provides distinct advantages over alternative tools:

### TensorBoard vs. SwanLab

*   **Cloud Support:** Easily sync training experiments to the cloud.
*   **Collaboration:** Facilitates teamwork and sharing within your organization.
*   **Dashboard:** Persistent and centralized dashboards for consistent results, regardless of the training environment.
*   **Robust Tables:**  Improved tables make it easy to search, filter, and compare results across experiments.

### Weights & Biases vs. SwanLab

* Weights & Biases is a closed-source, cloud-based MLOps platform.
* SwanLab offers both a cloud version, along with an open-source, free, and self-hosted option.

<br/>

## Community

### Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Documentation repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard web code
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting scripts

### Community & Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report issues and ask questions.
*   [Email Support](zeyi.lin@swanhub.co): Contact us for feedback.
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Discuss SwanLab and the latest AI tech.

### SwanLab README Badges

Add the SwanLab badge to your project's README:

[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![SwanLab Visualize](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Publications

If you find SwanLab helpful, please cite it:

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

Interested in contributing? Please review the [Contribution Guidelines](CONTRIBUTING.md).

We appreciate your support through social media, events, and conferences!

<br/>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br/>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## License

SwanLab is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

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

[demo-qwen2-grpo]:https://swanlab.cn/@kmno4/Qwen-R1/runs/t0zr3ak5r7188mjbjgdsc/chart
[demo-qwen2-grpo-image]: readme_files/example-qwen2-grpo.png

[tracking-swanlab-shield-link]:https://swanlab.cn
[tracking-swanlab-shield]: https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg

[visualize-swanlab-shield-link]:https://swanlab.cn
[visualize-swanlab-shield]: https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg

[dockerhub-shield]: https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square
[dockerhub-link]: https://hub.docker.com/r/swanlab/swanlab-next/tags