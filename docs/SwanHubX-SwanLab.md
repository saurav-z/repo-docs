<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: Open-Source Deep Learning Experiment Tracking and Visualization

SwanLab is a user-friendly, open-source tool that helps you effortlessly track, visualize, and collaborate on your deep learning experiments.  Improve your machine learning workflow with easy-to-use experiment tracking.

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

![](readme_files/swanlab-overview.png)

[中文 / English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features of SwanLab

*   **Experiment Tracking and Visualization:** Log metrics, hyperparameters, media (images, audio, text, video), and custom charts to visualize your training progress in real-time.
*   **Flexible Framework Integration:**  Seamlessly integrates with 30+ popular deep learning frameworks, including PyTorch, Hugging Face Transformers, and more.
*   **Hardware Monitoring:**  Monitor CPU, GPU (Nvidia, Ascend, etc.), memory, disk, and network utilization to understand resource usage.
*   **Cloud and Offline Support:** Use SwanLab online or offline with self-hosted options.
*   **Collaboration and Sharing:** Share experiments with your team, compare results, and collaborate on projects.
*   **Comprehensive Data Logging:** Automatically logs experiment metadata, including Git commit information, Python environment details, and hardware configurations.
*   **Rich Charting Capabilities:** Offers a wide range of chart types, including line charts, scatter plots, and more, for in-depth analysis.
*   **Customizable Plugins:**  Extend functionality with plugins for notifications, data writing, and more.
*   **Experiment Comparison:**  Compare hyperparameters and results across different experiments to identify winning configurations.

<br/>

## Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .
```

</details>

<details><summary>Offline Dashboard Installation</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login

1.  [Sign up](https://swanlab.cn) for a free account.
2.  Get API Key in settings ([API Key](https://swanlab.cn/settings)).
3.  Enter your API Key in the terminal.

```bash
swanlab login
```

### 3. Integrate into your Code

```python
import swanlab

# Initialize an experiment
swanlab.init(
    project="my-first-ml",
    config={'learning-rate': 0.003},
)

# Log metrics
for i in range(10):
    swanlab.log({"loss": i, "acc": i})
```

See your results on the [SwanLab](https://swanlab.cn) website!

<br/>

## Explore SwanLab

*   [**Online Demos**](https://docs.swanlab.cn/zh/examples/mnist.html):  Interactive examples demonstrating SwanLab's capabilities.
*   [**Self-Hosting Guide**](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html): Deploy your own SwanLab instance.
*   [**Framework Integrations**](https://docs.swanlab.cn/guide_cloud/integration/):  Learn how to integrate with your favorite frameworks.
*   [**Tutorials and Examples**](https://docs.swanlab.cn/zh/examples/):  Step-by-step guides and code examples.

## Resources

*   [**Documentation**](https://docs.swanlab.cn)
*   [**Report Issues**](https://github.com/swanhubx/swanlab/issues)
*   [**Feedback and Suggestions**](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc)
*   [**Changelog**](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html)
*   [**Community**](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)
*   [**License**](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)

## Community and Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report bugs and request features.
*   [Email Support](zeyi.lin@swanhub.co): Contact for support.
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discuss and get help from other users.

### Cite SwanLab

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

### Contribute

Contributions are welcome!  Please review the [CONTRIBUTING.md](CONTRIBUTING.md) file.

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>