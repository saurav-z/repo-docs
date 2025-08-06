<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## SwanLab: Open-Source Deep Learning Experiment Tracking & Visualization

**SwanLab is your all-in-one solution for effortlessly tracking, visualizing, and collaborating on your deep learning experiments, offering cloud and offline support and seamless integration with 30+ popular frameworks.**

[🔥 SwanLab Online](https://swanlab.cn) | [📃 Documentation](https://docs.swanlab.cn) | [🐛 Report an Issue](https://github.com/swanhubx/swanlab/issues) | [💬 Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) | [📜 Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) | [🤝 Community](https://github.com/SwanHubX/assets/blob/main/community.svg) | [📊 Benchmarks](https://swanlab.cn/benchmarks)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

<img src="readme_files/swanlab-overview.png" alt="SwanLab Overview" width="100%">

[中文 / English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking & Logging**:
    *   🚀 Track and visualize key metrics, hyperparameters, and experiment metadata.
    *   📝 Log scalars, images, audio, text, videos, 3D point clouds, and more.
    *   🔄  Supports over 30 popular frameworks (PyTorch, TensorFlow, etc.) for seamless integration.

*   **Intuitive Visualization**:
    *   📊  Interactive dashboards and charts for insightful data analysis.
    *   📈 Comprehensive chart types: Line charts, media charts, 3D point clouds, custom charts, and more.
    *   🔬 Customizable plots, including LLM-generated content visualization.

*   **Hardware Monitoring**:
    *   💻 Real-time monitoring of CPU, GPU (Nvidia, AMD, Intel, Apple Silicon), NPU (Ascend), MLU, XPU, and memory usage.
    *   💾 Monitor disk I/O and network activity.

*   **Collaboration & Management**:
    *   👥 Online collaboration for team-based experiments.
    *   📦 Centralized dashboard for efficient project and experiment management.
    *   🆚 Compare experiments side-by-side for insightful results analysis.
    *   ✉️ Share experiment results with persistent URLs.

*   **Flexibility & Extensibility**:
    *   ☁️ Supports cloud-based and offline usage.
    *   🔌 Plugin support to extend functionalities.
    *   💻 Self-hosting options.

*   **Framework Integrations**: Comprehensive integration with a wide range of machine learning frameworks.

*   **Comprehensive Features**: The core features that SwanLab provides.

<br/>

## Getting Started

1.  **Installation**:

    ```bash
    pip install swanlab
    ```

    <details><summary>Source Installation</summary>

    ```bash
    # Method 1
    git clone https://github.com/SwanHubX/SwanLab.git
    pip install -e .

    # Method 2
    pip install git+https://github.com/SwanHubX/SwanLab.git
    ```

    </details>

    <details><summary>Offline Dashboard Installation</summary>

    [Offline Dashboard Documentation](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)

    ```bash
    pip install 'swanlab[dashboard]'
    ```

    </details>

2.  **Sign up & Get API Key**:
    *   [Register for a free account](https://swanlab.cn).
    *   Log in to your account.
    *   Go to User Settings > [API Key](https://swanlab.cn/settings) and copy your API Key.
    *   Open your terminal and enter:

    ```bash
    swanlab login
    ```

    *   Enter your API Key when prompted and press Enter to log in.

3.  **Integrate SwanLab into Your Code**:

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

    Then visit [SwanLab](https://swanlab.cn) to view your experiments.

<br/>

## Self-Hosting

Self-host the community version and check out the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### Deploy with Docker

For detailed instructions, please refer to the [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Installation for China users:

```bash
./install.sh
```

Pull the image from DockerHub and install:

```bash
./install-dockerhub.sh
```

### Using the Self-Hosted Service

Login to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

You are now ready to log experiments to your self-hosted server.

<br/>

## Examples & Resources

Explore these resources to get started with SwanLab:

*   **Tutorials:**
    *   [MNIST Handwritten Digits Recognition](https://docs.swanlab.cn/examples/mnist.html)
    *   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
    *   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
    *   [ResNet Cat and Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
    *   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
    *   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
    *   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
    *   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
    *   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
    *   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)

<br/>

## Contributing

*   If you're interested in contributing to SwanLab, please start by reading the [Contribution Guidelines](CONTRIBUTING.md).

*   Share SwanLab on social media, at events, and in your papers to help us grow!

*   Consider citing SwanLab in your research:

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