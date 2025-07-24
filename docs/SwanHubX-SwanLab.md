<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br/>

## SwanLab: The Open-Source Deep Learning Experiment Tracker and Visualizer

**SwanLab** is an open-source, modern, and user-friendly tool for tracking, visualizing, and collaborating on your deep learning experiments, offering seamless integration with 30+ popular frameworks and both cloud and offline usage. 

[![][release-shield]][release-link]
[![][dockerhub-shield]][dockerhub-link]
[![][github-stars-shield]][github-stars-link]
[![][github-issues-shield]][github-issues-shield-link]
[![][github-contributors-shield]][github-contributors-link]
[![][license-shield]][license-shield-link]  
[![][tracking-swanlab-shield]][tracking-swanlab-shield-link]
[![][last-commit-shield]][last-commit-shield-link]
[![][pypi-version-shield]][pypi-version-shield-link]
[![][wechat-shield]][wechat-shield-link]
[![][pypi-downloads-shield]][pypi-downloads-shield-link]
[![][colab-shield]][colab-shield-link]

<div align="center">
<img src="readme_files/swanlab-overview.png" alt="SwanLab Overview" width="80%">
</div>

[中文 / English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

Join our [微信群](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Experiment Tracking & Visualization**: Easily track metrics, hyperparameters, and model outputs with an intuitive UI.

    *   **Cloud & Offline Support**: Use SwanLab online or offline, with data accessible from anywhere.
    *   **Framework Integrations**: Seamlessly integrates with 30+ leading deep learning frameworks.
    *   **Comprehensive Data Types**: Supports various data types, including scalars, images, audio, text, video, 3D point clouds, and custom ECharts charts.
    *   **Advanced Charting**: Offers a wide range of chart types, including line charts, media views (images, audio, text, video), 3D point clouds, and custom charts.
    *   **LLM Content Visualization**: Provides specialized text visualization for LLM training, supporting Markdown rendering.
    *   **Automated Logging**: Automatically logs metrics, hardware environment, Git repository info, Python environment, and project details.
    *   **Resume Training**: Support for resuming interrupted or completed training runs.
*   **Hardware Monitoring**: Monitor CPU, GPU (Nvidia, Ascend, Apple, Cambricon, Kunlunxin, Moore Threads, Metax, Hygon), and memory usage in real-time.
*   **Experiment Management**: Centralized dashboard for managing multiple projects and experiments.
*   **Result Comparison**: Compare hyperparameters and results across different experiments to discover insights.
*   **Collaboration**: Share and collaborate on experiments with your team, fostering discussion and insights.
*   **Easy Sharing**: Share experiment results with persistent URLs for easy dissemination.
*   **Self-Hosting**: Supports offline use for private dashboards and experiment management.
*   **Plugin Extensibility**: Extend functionality with plugins like [飞书通知](https://docs.swanlab.cn/plugin/notification-lark.html), [Slack通知](https://docs.swanlab.cn/plugin/notification-slack.html), [CSV记录器](https://docs.swanlab.cn/plugin/writer-csv.html), and more.

>   \[!IMPORTANT]
>
>   **Star the project** to receive all release notifications from GitHub without delay! ⭐️

<div align="center">
    <img src="readme_files/star-us.png" width="300">
</div>

<br>

## Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

For the latest features, install from source:

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Extension</summary>

[Offline Dashboard Documentation](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get Your API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Log in and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and run:

```bash
swanlab login
```

   Enter your API Key when prompted and press Enter to complete the login.

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

Congratulations! View your first SwanLab experiment at [SwanLab](https://swanlab.cn).

<br/>

## Self-Hosting

Self-hosting the community version allows you to view the SwanLab dashboard offline.

<div align="center">
<img src="./readme_files/swanlab-docker.png" alt="Swanlab Docker Deployment" width="60%">
</div>

### 1. Deploy Self-Hosted Version with Docker

Refer to the documentation for detailed instructions: [文档](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For quick installation in China:

```bash
./install.sh
```

Alternatively, pull the image from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Direct Experiments to Your Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, experiments will be recorded to your self-hosted service.

<br/>

## Practical Examples

**Open Source Projects using SwanLab:**

-   [happy-llm](https://github.com/datawhalechina/happy-llm): LLM tutorial ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
-   [self-llm](https://github.com/datawhalechina/self-llm): LLM tutorial ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
-   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series analysis and reproduction. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)

**Papers using SwanLab:**

-   [Animation Needs Attention](https://arxiv.org/abs/2507.03916)

**Tutorial Articles:**

-   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
-   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
-   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
-   [Resnet Cat Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
-   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
-   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
-   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
-   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
-   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
-   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
-   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
-   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
-   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
-   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
-   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
-   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
-   [Qwen2-VL Multimodal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
-   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
-   [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
-   [LeRobot Embodied AI Guide](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)

<br/>

## Hardware Monitoring

SwanLab tracks the hardware information and resource usage during AI training. Here is the support table:

| Hardware       | Information Recording | Resource Monitoring | Script                                                              |
| :------------- | :-------------------- | :------------------ | :------------------------------------------------------------------ |
| Nvidia GPU     | ✅                    | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU     | ✅                    | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC      | ✅                    | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU  | ✅                    | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU  | ✅                    | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU      | ✅                    | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU      | ✅                    | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU            | ✅                    | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)           |
| Memory         | ✅                    | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)        |
| Disk           | ✅                    | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)           |
| Network        | ✅                    | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)   |

If you wish to record other hardware, please submit an Issue or PR!

<br/>

## Framework Integrations

Combine your favorite frameworks with SwanLab!  
Here's a list of integrated frameworks; please submit an [Issue](https://github.com/swanhubx/swanlab/issues) to suggest frameworks you'd like integrated.

**Base Frameworks**

-   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
-   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
-   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Fine-tuning Frameworks**

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

<br/>

## Plugins and APIs

Extend SwanLab's capabilities with plugins to enhance your experiment management experience!

-   [Customize Your Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
-   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
-   [飞书通知](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
-   [企业微信通知](https://docs.swanlab.cn/plugin/notification-wxwork.html)
-   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
-   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
-   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
-   [FileLogdir Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open APIs:
- [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br/>

## Comparison with Similar Tools

### TensorBoard vs SwanLab

-   **Cloud-Based Support**: SwanLab facilitates easy online synchronization and saving of training experiments, enabling remote progress monitoring, experiment management, link sharing, real-time notifications, and multi-device access. TensorBoard is designed for offline experiment tracking.

-   **Collaboration Features**: SwanLab excels in supporting multi-person and cross-team machine learning collaborations, simplifying project management, experiment link sharing, and cross-space discussions. TensorBoard primarily serves individual use, lacking robust collaboration features.

-   **Persistent, Centralized Dashboards**: SwanLab stores results in a centralized dashboard regardless of your training environment (local machine, lab clusters, or cloud instances).  TensorBoard requires manual file management across different machines.

-   **Enhanced Table Views**: SwanLab's tables allow you to view, search, and filter results from diverse experiments, facilitating easy comparison of thousands of model versions to identify the best-performing models for your tasks. TensorBoard is not optimized for large-scale projects.

### Weights & Biases vs SwanLab

-   Weights & Biases is a closed-source MLOps platform that requires an internet connection.

-   SwanLab offers both internet-based and open-source, free, self-hosted options.

<br/>

## Community

### Related Repositories

-   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official Documentation
-   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline Dashboard Web Code
-   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting Scripts

### Community & Support

-   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Report issues and questions.
-   [Email Support](zeyi.lin@swanhub.co): For feedback on SwanLab.
-   [微信交流群](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Discussing SwanLab and sharing AI technology.

### SwanLab README Badges

Add a SwanLab badge to your README:

[![][tracking-swanlab-shield]][tracking-swanlab-shield-link] and [![][visualize-swanlab-shield]][visualize-swanlab-shield-link]

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Your Research

If SwanLab has been helpful for your research, consider citing it using the following format:

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

Interested in contributing to SwanLab? Please read the [Contribution Guide](CONTRIBUTING.md).

We also appreciate support via social media and events. Thank you!

<br/>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br/>

<img src="./readme_files/swanlab-and-user.png" width="50%" alt="SwanLab and User Image"/>

## License

This repository is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

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