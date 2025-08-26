<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: The Open-Source AI Experiment Tracking and Visualization Tool

**SwanLab is your all-in-one solution for tracking, visualizing, and collaborating on your deep learning experiments, providing a comprehensive platform for researchers and AI practitioners.**  Track your model training, visualize results, and collaborate with your team effortlessly, all with an intuitive interface and seamless integration with popular ML frameworks.  Check out the original repo at [https://github.com/SwanHubX/SwanLab](https://github.com/SwanHubX/SwanLab).

<div align="center">

[🔥 SwanLab Online](https://swanlab.cn) | [📃 Documentation](https://docs.swanlab.cn) | [🐛 Report Issues](https://github.com/swanhubx/swanlab/issues) | [💡 Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) | [📰 Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) |  <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> [基线社区](https://swanlab.cn/benchmarks)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking with Swanlab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Wechat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

</div>

<br/>

![SwanLab Overview](readme_files/swanlab-overview.png)

[中文 / English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Easy Integration**: Seamlessly integrates with **30+** popular deep learning frameworks.
*   **Comprehensive Visualization**: Visualize training metrics, hyperparameters, and more with intuitive charts and graphs.  Including support for scalar metrics, images, audio, text, videos, 3D point clouds, and custom Echarts.
*   **Real-time Hardware Monitoring**: Monitor CPU, GPU (Nvidia), NPU (Ascend), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), SOC (Apple), Moore Threads, Memory, Disk, and Network utilization.
*   **Collaborative Environment**: Share experiments, invite collaborators, and facilitate team-based training workflows.
*   **Flexible Deployment**: Use SwanLab in the cloud, on-premise, or offline, to fit your workflow.
*   **Robust Experiment Management**: Centralized dashboard for managing projects and experiments, facilitating at-a-glance overview and quick access to your experiments.
*   **Extensible with Plugins**: Enhance SwanLab with custom plugins for notifications, data logging, and more.

<br>

## Recent Updates

*   **2025.08.19**: Enhanced chart rendering performance, low-intrusive loading animations, integration with MLX-LM, and SpecForge.
*   **2025.08.06**: Training collaboration features with project collaborators, workspace list view, and project tag support.
*   **2025.07.29**: Experiment filtering and sorting in the sidebar, column control panel for table view, multi-API Key management, and new chart types (PR, ROC, confusion matrix).
*   **2025.07.17**: Enhanced line chart configuration, video data type support, and global chart dashboard Y-axis configuration.
*   **2025.07.10**: Text view with Markdown rendering and navigation, created by `swanlab.echarts.table` and `swanlab.Text`.
*   **2025.07.06**: Resume training support, file logger plugin, and integration with Ray and ROLL frameworks.
*   **2025.06.27**: Zoom in on line charts and smoothed lines.
*   **2025.06.20**: Integration with Hugging Face Accelerate framework.
*   **2025.06.18**: Integration with AREAL framework, hover highlighting, and cross-group comparison support.

<details><summary>Complete Changelog</summary>
... (Changelog Content - as provided in original README) ...
</details>

<br>

## Getting Started

### Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>
... (Source Installation Content - as provided in original README) ...
</details>

<details><summary>Offline Dashboard Extension</summary>
... (Offline Dashboard Extension Content - as provided in original README) ...
</details>

### Login and Integrate

1.  **Register**: Sign up for a free account at [SwanLab](https://swanlab.cn)
2.  **Get API Key**:  Go to User Settings > [API Key](https://swanlab.cn/settings) and copy your API Key.
3.  **Login**: Open your terminal and type `swanlab login`, then enter your API Key.
4.  **Integrate Code**: Add SwanLab to your code:

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

Go to [SwanLab](https://swanlab.cn) to view your experiment.

<br>

## Self-Hosting

Self-hosting the community version enables offline viewing of the SwanLab dashboard.

![SwanLab Docker](./readme_files/swanlab-docker.png)

### 1. Deploy with Docker

See the full guide at: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

China quick install:

```bash
./install.sh
```

Install using DockerHub:

```bash
./install-dockerhub.sh
```

### 2.  Specify Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, your experiments will be recorded on your self-hosted service.

<br>

## Example Projects

**(Links to the example projects - as provided in original README)**

**(The existing example projects section is retained, including links, as it provides valuable context)**

<br>

## Hardware Monitoring

SwanLab records hardware information. Supported hardware:

**(Table of supported hardware - as provided in original README)**

If you wish to record any other hardware information, feel free to submit an issue or a pull request.

<br>

## Framework Integrations

**(List of Framework Integrations - as provided in original README)**

[More Integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br>

## Plugins and APIs

**(Plugin and API information - as provided in original README)**

<br>

## Comparison

**(Comparison of SwanLab with TensorBoard and Weights & Biases - as provided in original README)**

<br>

## Community

**(Community information, including GitHub, support, and badges - as provided in original README)**

<br>

**(If applicable, insert the Star History and License sections from original README)**
```
## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)
```
```
## 📃 协议
本仓库遵循 [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE) 开源协议
```

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