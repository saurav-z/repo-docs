<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## SwanLab: Unleash the Power of AI Experiment Tracking and Visualization

**SwanLab is an open-source, user-friendly tool designed to track, visualize, and collaborate on your machine learning experiments.**  Seamlessly integrated with 30+ popular frameworks, SwanLab empowers researchers to gain insights, compare results, and streamline their AI workflows.  [Visit the original repo!](https://github.com/SwanHubX/SwanLab)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![SwanLab Cloud](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

<img src="readme_files/swanlab-overview.png" alt="SwanLab Overview">

中文 / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking and Visualization:**  Track key metrics, visualize training progress with intuitive charts, and monitor experiment performance in real-time.  Supports scalar metrics, images, audio, text, video, 3D point clouds, and more.

    *   **Rich Charting:** Create insightful visualizations with line charts, media views, custom ECharts, and specialized LLM content rendering.
    *   **Automatic Logging:**  Effortlessly capture logs, hardware information, Git repository details, Python environment, and project directory structure.
    *   **Resume Support:**  Continue training from checkpoints and seamlessly add new metrics to existing experiments.

*   **Framework Integrations:**  Integrate with 30+ popular machine learning frameworks including PyTorch, Hugging Face Transformers, PyTorch Lightning, and more.

*   **Hardware Monitoring:** Monitor CPU, NPU (Ascend), GPU (Nvidia, MetaX, Moore Threads), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), memory, disk, and network utilization in real-time.

*   **Experiment Management:** Organize and manage experiments with a centralized dashboard, allowing you to quickly view project overviews and track progress.

*   **Result Comparison:**  Compare experiment results using interactive tables and charts to identify trends and optimize your models.

*   **Online Collaboration:**  Enable collaborative training by syncing experiments within a project, allowing teams to share results, provide feedback, and enhance training efficiency.

*   **Result Sharing:**  Share your experiments easily with persistent URLs that can be sent to collaborators or embedded in online documentation.

*   **Self-Hosting Support:** Utilize the offline mode or self-hosted community version to access the dashboard and manage experiments, even without an internet connection.

*   **Plugin Extensibility:**  Expand SwanLab's functionality using plugins for features such as email notifications, Slack, and CSV logging.

<br>

## Getting Started

### 1. Installation

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

<details><summary>Offline Dashboard Extension Installation</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Log in & Get API Key

1.  [Sign up](https://swanlab.cn) for a free account.
2.  Log in and copy your API key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and run:

```bash
swanlab login
```

Enter your API key when prompted.

### 3. Integrate SwanLab into your code

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

Now, visit [SwanLab](https://swanlab.cn) to view your first experiment.

<br>

## Self-Hosting

Self-hosting the community version enables offline use of the SwanLab dashboard.

<img src="./readme_files/swanlab-docker.png" alt="SwanLab Docker Deployment">

### 1. Deploying with Docker

For deployment, refer to the [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker

# For China-based users, install:
./install.sh

# Install from DockerHub:
./install-dockerhub.sh
```

### 2. Pointing Experiments to Your Self-Hosted Instance

Log into your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Once logged in, all experiment logs will be recorded to your self-hosted service.

<br>

## Examples & Tutorials

*   **[ResNet50 Cat/Dog Classification](https://swanlab.cn/@ZeyiLin/Cats_Dogs_Classification/runs/jzo93k112f15pmx14vtxf/chart)**
*   **[YOLOv8 COCO128 Object Detection](https://swanlab.cn/@ZeyiLin/ultratest/runs/yux7vclmsmmsar9ear7u5/chart)**
*   **[Qwen2 Instruction Fine-tuning](https://swanlab.cn/@ZeyiLin/Qwen2-fintune/runs/cfg5f8dzkp6vouxzaxlx6/chart)**
*   **[LSTM Google Stock Prediction](https://swanlab.cn/@ZeyiLin/Google-Stock-Prediction/charts)**

[More examples](https://docs.swanlab.cn/zh/examples/mnist.html)

🌟 If you have a tutorial to share, create a PR!

<br>

## Hardware Recording

SwanLab records hardware information and resource usage during AI training. Here's the supported hardware:

| Hardware         | Information Recording | Resource Monitoring | Script                                                                                |
| ---------------- | --------------------- | ------------------- | ------------------------------------------------------------------------------------- |
| Nvidia GPU       | ✅                    | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU       | ✅                    | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC        | ✅                    | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU    | ✅                    | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU    | ✅                    | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU| ✅                    | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU        | ✅                    | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU        | ✅                    | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU              | ✅                    | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)  |
| Memory           | ✅                    | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk             | ✅                    | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network          | ✅                    | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Contribute by submitting Issues and PRs if you want to add support for other hardware.

<br>

## Framework Integrations

Easily integrate your favorite frameworks with SwanLab!  Here's a list of integrated frameworks:

**Basic Frameworks**

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Fine-tuning Frameworks**

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

[More integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br>

## Plugins & API

Enhance your experiment management with SwanLab plugins!

*   [Create your own plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Directory Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:

*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparisons

### Tensorboard vs. SwanLab

-   **Cloud Support**: SwanLab enables easy cloud synchronization and saving of training experiments, facilitating remote monitoring, project management, experiment sharing, real-time notifications, and multi-device access. Tensorboard is primarily a local experiment-tracking tool.
-   **Collaboration**: SwanLab streamlines collaboration, providing tools for managing training projects, sharing experiments, and team communication. Tensorboard is mainly designed for individuals.
-   **Dashboard**: SwanLab's centralized dashboard stores results, regardless of where your model trains, allowing for easier management. Tensorboard requires manual copying and management of TFEvent files.
-   **Enhanced Tables**: SwanLab's tables facilitate viewing, searching, and filtering results, simplifying the process of analyzing thousands of model versions. Tensorboard is not ideal for large projects.

### Weights and Biases vs. SwanLab

*   Weights and Biases is a closed-source, cloud-based MLOps platform.
*   SwanLab supports cloud usage but offers an open-source, free, and self-hosted option.

<br>

## Community

### Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard repository
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting scripts repository

### Community Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report errors and issues.
*   [Email Support](zeyi.lin@swanhub.co): For feedback and questions about using SwanLab.
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): For discussing SwanLab and the latest AI technologies.

### SwanLab README Badges

Add a SwanLab badge to your README:

[![SwanLab Cloud](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![SwanLab Cloud](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

More design assets can be found at: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab

If you use SwanLab, please consider citing it in your research:

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

Review the [Contribution Guide](CONTRIBUTING.md).  Share SwanLab on social media, at events and conferences!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" alt="SwanLab and Users">

## License

SwanLab is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)