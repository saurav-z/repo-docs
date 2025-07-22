<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br>

## SwanLab: Open-Source Deep Learning Experiment Tracking and Visualization

**SwanLab is an open-source, modern tool for tracking, visualizing, and collaborating on your deep learning experiments, making it easy to monitor and analyze your model training.** Built for researchers and AI developers, SwanLab offers a user-friendly interface and robust features to streamline your machine learning workflow.  [Check out the original repo](https://github.com/SwanHubX/SwanLab).

<div align="center">
  <a href="https://swanlab.cn">🔥 SwanLab Online</a> · <a href="https://docs.swanlab.cn">📃 Documentation</a> · <a href="https://github.com/swanhubx/swanlab/issues">Report Issues</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">Feedback</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">Changelog</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">Benchmarks</a>
</div>

[![GitHub Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![Docker Image Version](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

<div align="center">
  <img src="readme_files/swanlab-overview.png" alt="SwanLab Overview" width="600">
</div>

<div align="center">
  <br>
  中文 / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)
  <br>
  👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)
  <br>
  <a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank">
    <img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" />
  </a>
</div>

<br/>

## Key Features

*   **Experiment Tracking and Visualization:** Easily track and visualize key metrics, hyperparameters, and model outputs during training.
    *   **Real-time monitoring:** See your model's performance as it trains.
    *   **Rich Visualization:** Visualize your data with line charts, media views, and more
    *   **Supports various data types:** Scalar values, images, audio, text, video, 3D point clouds, molecular data, and custom Echarts charts.

*   **Flexible Integration:** Seamlessly integrate with 30+ popular deep learning frameworks, including:
    *   PyTorch, TensorFlow, Hugging Face Transformers, PyTorch Lightning, and more.  See [Supported Frameworks](#-framework-integration)
*   **Hardware Monitoring:** Monitor CPU, GPU, Ascend NPU, Apple SOC, MLU, XPU, DCU, Moore Threads, and Memory usage.
*   **Experiment Management:** Organize and manage multiple projects and experiments through a centralized dashboard.
*   **Result Comparison:** Compare hyperparameters and results across different experiments to identify trends and insights.
*   **Collaboration:** Collaborate with your team in real-time and share results via persistent URLs.
*   **Self-Hosting:** Deploy and use SwanLab offline or on your own servers.
*   **Plugin Extensibility:** Extend SwanLab's functionality with plugins for notifications, data logging, and more.

## Getting Started

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .
```

or

```bash
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
2.  Log in and find your API Key in User Settings > [API Key](https://swanlab.cn/settings).
3.  Open a terminal and run:

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

Visit [SwanLab](https://swanlab.cn) to see your experiment.

<br>

## 💻 Self-Hosting

Self-hosting allows you to use SwanLab offline.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy a Self-Hosted Version with Docker

See the [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html) for instructions.

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Quick install for China:

```bash
./install.sh
```

Install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Direct Experiments to the Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, your experiments will be recorded to the self-hosted service.

<br>

## 🔥 Real-World Examples

SwanLab provides a growing collection of examples, tutorials, and integrations.

*   **Tutorials:**  A variety of tutorials are available in the documentation, covering tasks like image classification, object detection, and more.  See [Tutorials](#-实战案例)
*   **Open Source Projects:**  Check out some open-source projects using SwanLab to enhance your machine learning workflows. See [Tutorials](#-实战案例)

<br>

## 🎮 Hardware Monitoring

SwanLab tracks hardware information and resource usage during AI training:

| Hardware        | Information Tracking | Resource Monitoring | Script                                                                       |
| --------------- | ------------------- | ------------------- | ---------------------------------------------------------------------------- |
| NVIDIA GPU      | ✅                   | ✅                   | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)       |
| Ascend NPU      | ✅                   | ✅                   | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)       |
| Apple SOC       | ✅                   | ✅                   | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)       |
| Cambricon MLU   | ✅                   | ✅                   | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py)  |
| Kunlunxin XPU   | ✅                   | ✅                   | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py)  |
| Moore Threads GPU | ✅                   | ✅                   | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU        | ✅                   | ✅                   | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU        | ✅                   | ✅                   | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU             | ✅                   | ✅                   | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)             |
| Memory          | ✅                   | ✅                   | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)          |
| Disk            | ✅                   | ✅                   | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)            |
| Network         | ✅                   | ✅                   | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)       |

<br>

## 🚗 Framework Integration

Integrate your favorite frameworks with SwanLab!

**Base Frameworks**
*   PyTorch
*   MindSpore
*   Keras

**Specialized/Fine-tuning Frameworks**
*   PyTorch Lightning
*   HuggingFace Transformers
*   LLaMA Factory
*   Modelscope Swift
*   DiffSynth Studio
*   Sentence Transformers
*   PaddleNLP
*   OpenMind
*   Torchtune
*   XTuner
*   MMEngine
*   FastAI
*   LightGBM
*   XGBoost

**Evaluation Frameworks**
*   EvalScope

**Computer Vision**
*   Ultralytics
*   MMDetection
*   MMSegmentation
*   PaddleDetection
*   PaddleYOLO

**Reinforcement Learning**
*   Stable Baseline3
*   veRL
*   HuggingFace trl
*   EasyR1
*   AReaL
*   ROLL

**Other Frameworks:**
*   Tensorboard
*   Weights&Biases
*   MLFlow
*   HuggingFace Accelerate
*   Ray
*   Unsloth
*   Hydra
*   Omegaconf
*   OpenAI
*   ZhipuAI

[More Integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br>

## 🔌 Plugins and API

Extend SwanLab's capabilities with plugins and open APIs!

-   [Customize Your Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
-   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [Enterprise WeChat Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [File Log Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open APIs:
-   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparison with Similar Tools

### Tensorboard vs SwanLab

*   **Online Support:** SwanLab offers seamless online synchronization and saving of training experiments to the cloud, enabling easy remote monitoring.  Tensorboard is a local experiment tracking tool.

*   **Collaboration:** SwanLab allows for collaborative machine learning projects, easy project management, experimental result sharing. Tensorboard primarily serves individual needs and is not designed for collaboration.

*   **Centralized Dashboard:** SwanLab results are recorded in a centralized dashboard.  TensorBoard requires management of TFEvent files from different machines, which can be time consuming.

*   **Powerful Tables:** SwanLab tables enable viewing, searching, and filtering results from different experiments. TensorBoard is not well-suited for large projects.

### Weights and Biases vs SwanLab

*   Weights and Biases is a closed-source MLOps platform that requires a network connection.
*   SwanLab supports cloud services, and also offers open-source, free, and self-hosted versions.

<br>

## 👥 Community

### Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation.
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard web code.
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting deployment scripts.

### Community and Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report errors and issues.
*   [Email Support](zeyi.lin@swanhub.co): Provide feedback.
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Discuss SwanLab.

### SwanLab README Badges

Add a SwanLab badge to your README:

[![Track with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![Track with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More assets available in the [assets](https://github.com/SwanHubX/assets) repository.

### Citing SwanLab

If SwanLab has been helpful for your research, please cite us:

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

Read the [Contribution Guidelines](CONTRIBUTING.md) and consider contributing!

We also appreciate sharing, supporting, and contributing to SwanLab!

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

[demo-qwen2-grpo]:https://swanlab.cn/@kmno4/Qwen-R1/runs/t0zr3ak5r7188mjbjgdsc/chart
[demo-qwen2-grpo-image]: readme_files/example-qwen2-grpo.png

[tracking-swanlab-shield-link]:https://swanlab.cn
[tracking-swanlab-shield]: https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg

[visualize-swanlab-shield-link]:https://swanlab.cn
[visualize-swanlab-shield]: https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg

[dockerhub-shield]: https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square
[dockerhub-link]: https://hub.docker.com/r/swanlab/swanlab-next/tags