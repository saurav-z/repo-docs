<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## SwanLab: Supercharge Your Machine Learning Workflow with Open-Source Experiment Tracking and Visualization

**SwanLab** is a powerful, open-source tool designed to revolutionize how you track, visualize, and collaborate on your deep learning experiments, offering a modern and user-friendly interface to accelerate your research and development.  Easily integrate with 30+ frameworks and choose between cloud and self-hosted options.

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

[<img src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg">](https://swanlab.cn)
[<img src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg">](https://swanlab.cn)

<br/>

<div align="center">
<img src="readme_files/swanlab-overview.png" alt="SwanLab Overview" width="80%">
</div>

<br/>

**Quick Links:** [🔥 Online Demo](https://swanlab.cn) | [📃 Documentation](https://docs.swanlab.cn) | [🙋‍♀️ Issues](https://github.com/swanhubx/swanlab/issues) | [🙏 Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) | [🔄 Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) | [🤝 Community](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

[中文 / English / 日本語 / Русский](README_EN.md)

<br/>

## Key Features

*   **Experiment Tracking & Visualization:**
    *   Track and visualize key metrics, hyperparameters, and model outputs with a clean, intuitive UI.
    *   Supports a wide range of data types:  scalar metrics, images, audio, text, videos, 3D point clouds, biochemical molecules, and custom ECharts charts.
    *   **Comprehensive Charting:** Line charts, media charts (images, audio, text, videos), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, and [custom charts](https://docs.swanlab.cn/guide_cloud/experiment_track/log-custom-chart.html).
    *   **LLM-Specific Visualization:** Text visualization components designed for LLM training scenarios, with Markdown rendering support.

        <div align="center">
            <img src="readme_files/text-chart.gif" alt="Text Chart" width="60%">
        </div>

*   **Framework Integrations:**
    *   Seamless integration with **30+ popular machine learning frameworks**, including PyTorch, Hugging Face Transformers, PyTorch Lightning, LLaMA Factory, and many more.
    *   Expand integrations using the following [frameworks](https://docs.swanlab.cn/guide_cloud/integration/)

        <div align="center">
            <img src="readme_files/integrations.png" alt="Framework Integrations" width="70%">
        </div>

*   **Hardware Monitoring:**
    *   Real-time monitoring of hardware resources, including CPU, NPU, GPU (Nvidia, Ascend, Cambricon, Kunlunxin, Moore Threads, Metax, Hygon), memory, and disk utilization.

*   **Experiment Management & Comparison:**
    *   Centralized dashboard for managing multiple projects and experiments.
    *   Compare experiments side-by-side with interactive tables and charts to analyze results.

    <div align="center">
        <img src="readme_files/swanlab-table.png" alt="SwanLab Table" width="80%">
    </div>

*   **Collaboration & Sharing:**
    *   Collaborate with your team by sharing experiments and results.
    *   Generate shareable, persistent URLs for individual experiments.

*   **Self-Hosting:**
    *   Run SwanLab locally or on your own servers, with community-supported self-hosting options for offline use.

*   **Extensibility:**
    *   Plugin architecture for extending SwanLab's functionality with custom features.

## Getting Started

1.  **Install:**
    ```bash
    pip install swanlab
    ```

    <details>
        <summary>Source Installation</summary>

        ```bash
        git clone https://github.com/SwanHubX/SwanLab.git
        pip install -e .
        ```
    </details>

2.  **Login:**
    *   Register for a free account on [SwanLab](https://swanlab.cn).
    *   Go to your user settings and copy your API Key.
    *   Run `swanlab login` in your terminal and enter your API Key.

3.  **Integrate with your code:**

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

    Then go to [SwanLab](https://swanlab.cn) to view your experiments!

## Use Cases and Examples

See our [Example](https://docs.swanlab.cn/zh/examples/mnist.html) for a demonstration.

*   **Tutorials:**
    *   [MNIST Handwritten Digits](https://docs.swanlab.cn/examples/mnist.html)
    *   [FashionMNIST Image Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
    *   [CIFAR10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
    *   [ResNet Cat/Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)

*   **Example Projects**
    *   [happy-llm](https://github.com/datawhalechina/happy-llm)
    *   [self-llm](https://github.com/datawhalechina/self-llm)
    *   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek)
    *   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL)

[More Examples](https://docs.swanlab.cn/zh/examples/)

## Self-Hosting

Self-host the community edition for offline access to the SwanLab dashboard.

<img src="./readme_files/swanlab-docker.png" width="70%" />

1.  **Deploy with Docker:**  Follow the instructions in the [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

    ```bash
    git clone https://github.com/SwanHubX/self-hosted.git
    cd self-hosted/docker
    ./install.sh  # For China users
    # OR
    ./install-dockerhub.sh # for DockerHub
    ```

2.  **Login to your self-hosted service:**

    ```bash
    swanlab login --host http://localhost:8000
    ```
    After logging in, your experiments will be recorded on your self-hosted service.

## Community

*   **Join the Community:**
    *   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues) - Report issues and ask questions.
    *   [Email Support](zeyi.lin@swanhub.co) - Get in touch with SwanLab developers
    *   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html) - Connect with other users and the SwanLab team.

*   **Contribute:**  Read the [Contribution Guidelines](CONTRIBUTING.md).

*   **Cite SwanLab:**

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

*   **Share SwanLab**:  Add the SwanLab badge to your README to help promote SwanLab.

    ```
    [![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
    [![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
    ```

    More Design Materials: [assets](https://github.com/SwanHubX/assets)

## License

SwanLab is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)