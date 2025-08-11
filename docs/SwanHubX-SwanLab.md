<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

##  SwanLab: Your Open-Source Toolkit for Deep Learning Experiment Tracking and Visualization

**SwanLab empowers AI researchers with an intuitive platform to track, visualize, and collaborate on deep learning experiments, whether you're working locally or in the cloud.**  It seamlessly integrates with 30+ popular frameworks.

[🔥SwanLab Online](https://swanlab.cn) · [📃 Documentation](https://docs.swanlab.cn) · [🐞 Report an Issue](https://github.com/swanhubx/swanlab/issues) · [💡 Suggest Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) · [📜 Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) ·  [🤝 Community](https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
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

<br/>

[<img src="readme_files/swanlab-overview.png" width="100%" />](https://swanlab.cn)

[English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md) / [中文](README.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

## Key Features of SwanLab

*   **Experiment Tracking and Visualization**:

    *   Track and visualize key metrics, hyperparameters, and experiment metadata.
    *   Visualize your training progress with an intuitive UI.
    *   Support for various data types: scalar metrics, images, audio, text, video, 3D point clouds, biochemical molecules, and custom ECharts charts.
    *   Create insightful visualizations with different chart types, including line charts, media charts, 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, and custom charts.
    *   Powerful text chart support, including markdown rendering.

    [![swanlab-echarts](readme_files/echarts.png)](https://swanlab.cn/@ZeyiLin/swanlab-echarts-demo/charts)

*   **Flexible Framework Integration**:

    *   Seamless integration with a wide range of deep learning frameworks, including PyTorch, Hugging Face Transformers, PyTorch Lightning, LLaMA Factory, and many more (30+). See [Framework Integration](#framework-integration).

    *  **Hardware Monitoring:**
        * Support real-time monitoring of CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (MXX), Moore Threads GPU (MTGPU), memory, disk I/O, and network.
        *   Complete hardware information collection and analysis.

*   **Experiment Management**:

    *   A centralized dashboard designed for training scenarios.
    *   Manage multiple projects and experiments with a global view.
    *   Quickly browse and analyze your experiments.

*   **Result Comparison**:

    *   Compare hyperparameters and results across different experiments with online tables and comparison charts.
    *   Discover inspiration for your iterations.

*   **Collaboration**:

    *   Enable collaborative training within your team.
    *   Synchronize experiments in real time within a project.
    *   View training records, share ideas, and provide suggestions.

*   **Sharing**:

    *   Share experiments easily by copying and sending persistent URLs.
    *   Embed experiments into online notes.

*   **Self-Hosted Deployment**:

    *   Support for offline environments.
    *   View dashboards and manage experiments with the self-hosted community version.

*   **Plugin Extensions**:

    *   Extend SwanLab's functionality with plugins, such as email notification, Feishu notification, Discord notification, Slack notification, CSV writer, and FileLogDir writer.
    *   Customize plugins to suit your specific needs.

> \[!IMPORTANT]
>
> **Star the project** to stay updated on releases! ⭐️

![star-us](readme_files/star-us.png)

<br/>

##  Key Benefits

*   **Increased Efficiency**:  Spend less time debugging and more time iterating on your models.
*   **Improved Collaboration**:  Streamline team communication and knowledge sharing.
*   **Better Insights**:  Gain a deeper understanding of your training process.

## Getting Started

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .
```
</details>

<details><summary>Install Offline Dashboard Extension</summary>

```bash
pip install 'swanlab[dashboard]'
```
</details>

### 2. Login & API Key
1.  [Sign up](https://swanlab.cn) for a free account.
2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  In your terminal:
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

Now go to [SwanLab](https://swanlab.cn) to view your first experiment!

<br/>

##  Self-Hosting

The self-hosted community version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploying Self-Hosted with Docker

Please refer to the documentation: [Docker Deployment Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For quick installation in China:

```bash
./install.sh
```

To install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2.  Point Experiments to Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Once logged in, all experiments will be logged to your self-hosted service.

<br/>

##  Real-World Examples

Explore tutorials and projects using SwanLab: See [Examples](https://docs.swanlab.cn/zh/examples/mnist.html) for more tutorials.

🌟 Contribute and submit PRs with the tutorial you want to be included!

<br/>

##  Hardware Monitoring
*   Provides real-time monitoring and logging of system hardware information during AI training.
*   Supports CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XLU (Kunlunxin), DCU (Hygon), MetaX GPU (MXX), Moore Threads GPU (MTGPU), memory, disk and network hardware indicators

##  Framework Integration

Connect SwanLab with your preferred frameworks!

Here's a list of integrated frameworks.  Submit a [GitHub Issue](https://github.com/swanhubx/swanlab/issues) for frameworks you'd like to see integrated.

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

[More integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br/>

##  Plugins and API

Extend SwanLab with plugins for a more enhanced experiment management experience!

*   [Customize your plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WxWork Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [FileLogDir Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br/>

##  Comparisons

### Tensorboard vs SwanLab

*   **Cloud Support**: SwanLab allows for easy cloud-based synchronization and saving of training experiments, enabling remote viewing, historical project management, and sharing experiment links. Tensorboard is an offline experiment tracking tool.

*   **Collaboration**: SwanLab excels in collaborative machine learning, simplifying the management of multi-person training projects, experiment sharing, and cross-team discussions. Tensorboard is primarily designed for individual use and lacks robust collaboration features.

*   **Persistent, Centralized Dashboard**:  Your results are logged to a single dashboard, whether you're training locally, in a lab, or in the cloud. TensorBoard requires manual management of TFEvent files.

*   **Powerful Tables**: SwanLab provides tables to view, search, and filter results from different experiments. Tensorboard has limitations in managing large projects.

### Weights and Biases vs SwanLab

*  SwanLab offers a free, open-source, self-hosted option, while Weights and Biases requires network access and is a closed-source MLOps platform.

<br/>

##  Community

###  Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Lightweight offline dashboard
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosted deployment scripts

### Community & Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report issues and ask questions.
*   [Email Support](zeyi.lin@swanhub.co): Contact us with questions regarding using SwanLab
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Get support and share your experiences.

### SwanLab README Badges

If you like using SwanLab, add the SwanLab badge to your README:

[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Publications

If you find SwanLab helpful in your research, please consider citing it:

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

Want to contribute?  Read the [contribution guidelines](CONTRIBUTING.md).  We welcome support through social media, events, and conferences!

<br/>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br/>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

##  License

This project is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)