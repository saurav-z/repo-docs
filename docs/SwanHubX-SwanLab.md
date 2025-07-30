<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

**SwanLab: Unleash the power of your AI experiments with a user-friendly, open-source platform that simplifies tracking, visualizing, and collaborating on your deep learning projects.**

<a href="https://swanlab.cn">🔥SwanLab 在线版</a> · <a href="https://docs.swanlab.cn">📃 文档</a> · <a href="https://github.com/swanhubx/swanlab/issues">报告问题</a> · <a href="https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc">建议反馈</a> · <a href="https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html">更新日志</a> · <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> <a href="https://swanlab.cn/benchmarks">基线社区</a>

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


![](readme_files/swanlab-overview.png)

中文 / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 加入我们的[微信群](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>

<br/>

## Key Features of SwanLab: Accelerate Your AI Research

SwanLab is a powerful, open-source tool designed to streamline your machine learning workflow. It simplifies experiment tracking, visualization, and collaboration, ultimately accelerating your research.  [Access the original repository here](https://github.com/SwanHubX/SwanLab).

*   **📊 Comprehensive Experiment Tracking:** Easily track and visualize key metrics, hyperparameters, and model artifacts.
    *   Cloud Support (similar to Weights & Biases) for remote access and collaboration.
    *   Comprehensive Metric Logging: scalars, images, audio, text, videos, 3D point clouds, molecules, and customizable charts.
    *   Robust Charting: Line plots, media views, 3D point clouds, molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, and custom charts.
    *   Advanced Text Visualization: Text components with Markdown rendering for LLM-generated content visualization.
    *   Automatic Logging: Hardware resources, Git information, Python environment, and project directories are automatically tracked.
    *   Resume Training: Add new metric data to an experiment after the training is completed or interrupted.
*   **⚡️ Extensive Framework Integration:** Seamlessly integrates with over 30 popular deep learning frameworks, including PyTorch, TensorFlow, Hugging Face Transformers, and many more.
*   **💻 Hardware Monitoring:** Real-time monitoring of CPU, GPU (Nvidia, Ascend, etc.), and memory usage.
*   **📦 Experiment Management:** A centralized dashboard for efficient project and experiment management.
*   **🆚 Result Comparison:** Compare experiments with a table and charts to identify insights and iterate effectively.
*   **👥 Collaborative Features:** Enable team collaboration with real-time experiment synchronization and shared access.
*   **✉️ Easy Sharing:** Generate and share persistent experiment URLs for easy collaboration and knowledge sharing.
*   **💻 Self-Hosting Support:** Run SwanLab offline for local experimentation and data privacy.
*   **🔌 Plugin Extensibility:** Extend functionality with plugins for features like email, notification (e.g., Slack, Discord, Lark) and CSV logging.

> \[!IMPORTANT]
>
> **Star the project** and stay updated on new releases! ⭐️

![star-us](readme_files/star-us.png)

<br>

## 📃 Online Demos
Experience SwanLab's capabilities with these interactive online demos:

| Demo                               | Description                                                              |
| ---------------------------------- | ------------------------------------------------------------------------ |
| [ResNet50 猫狗分类][demo-cats-dogs] | Tracks a simple ResNet50 model trained on a cat vs dog image classification dataset.   |
| [Yolov8-COCO128 目标检测][demo-yolo] | Tracks training hyperparameters and metrics using Yolov8 on the COCO128 dataset. |
| [Qwen2 指令微调][demo-qwen2-sft]  | Track Qwen2 large language model fine-tuning for instruction following.    |
| [LSTM Google 股票预测][demo-google-stock]   | Train an LSTM model to predict future stock prices using Google stock data. |
| [ResNeXt101 音频分类][demo-audio-classification] |  Track the progress of experiments and compare different models from ResNet to ResNeXt in audio classification tasks.   |
| [Qwen2-VL COCO数据集微调][demo-qwen2-vl] | Based on the Qwen2-VL multi-modal large model, perform LoRA fine-tuning on the COCO2014 dataset. |
| [EasyR1 多模态LLM RL训练][demo-easyr1-rl] | Perform multimodal LLM RL training using the EasyR1 framework  |
| [Qwen2.5-0.5B GRPO训练][demo-qwen2-grpo] | Training GRPO on the GSM8k dataset using the Qwen2.5-0.5B model. |
[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🏁 Quickstart: Get Started with SwanLab

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>
Install from source for latest features:

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```
</details>

<details><summary>Offline Dashboard Installation</summary>
Install for offline dashboard:

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

    Enter your API Key when prompted.

### 3. Integrate SwanLab into Your Code

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

That's it!  View your first SwanLab experiment at [SwanLab](https://swanlab.cn).

<br>

## 💻 Self-Hosting

The community edition allows you to view the SwanLab dashboard offline.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted Version with Docker

See the documentation at: [Docker Deployment Guide](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For fast installation in China:

```bash
./install.sh
```

For installation from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Direct Experiments to Your Self-Hosted Service

Log in to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

Then all the experiment logs are recorded to self-hosted service.

<br>

## 🔥 Real-World Examples
See how SwanLab is being used in the AI community:

**Tutorials and Open Source Projects:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): Tutorial on large language models.
*   [self-llm](https://github.com/datawhalechina/self-llm): Tutorial on fine-tuning LLMs.
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series.

**Academic Publications:**

*   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
*   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)

**Step-by-Step Tutorials:**
*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat vs. Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Cartpole Reinforcement Learning](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pretraining](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multi-Modal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multi-modal model training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied AI Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)

🌟 Submit a PR to add your tutorial!

<br>

## 🎮 Hardware Monitoring

SwanLab provides hardware information and resource usage during the training process.

| Hardware        | Information Recording | Resource Monitoring | Script                                                   |
| --------------- | --------------------- | ------------------- | -------------------------------------------------------- |
| Nvidia GPU      | ✅                    | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)    |
| Ascend NPU      | ✅                    | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)    |
| Apple SOC       | ✅                    | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)     |
| Cambricon MLU   | ✅                    | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU   | ✅                    | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅                  | ✅                  | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU      | ✅                  | ✅                  | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU      | ✅                  | ✅                  | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU             | ✅                    | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)       |
| Memory          | ✅                    | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)    |
| Disk            | ✅                    | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)     |
| Network         | ✅                    | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

Report an Issue or create a PR to record other hardware metrics.

<br>

## 🚗 Framework Integration

SwanLab seamlessly integrates with your favorite frameworks.  If you'd like to request an integration, please create an [Issue](https://github.com/swanhubx/swanlab/issues).

**Base Frameworks:**
*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Fine-tuning Frameworks:**
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

**Evaluation Frameworks:**
*   [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)

**Computer Vision:**
*   [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
*   [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
*   [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
*   [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
*   [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)

**Reinforcement Learning:**
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

<br>

## 🔌 Plugins and APIs

Enhance your experiment management with SwanLab plugins!

*   [Create Your Own Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [Dingtalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Work Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Log Directory Writer](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## 🆚 Comparison with Similar Tools

SwanLab stands out with features for experiment tracking, collaboration and flexibility:

### Tensorboard vs SwanLab

*   **☁️ Online Support:** SwanLab supports online syncing and saving experiments to the cloud. This allows for convenient remote training progress monitoring, project management, experiment link sharing, notifications, and multi-device access. TensorBoard is a local-only experiment tracking tool.

*   **👥 Multi-user Collaboration:** SwanLab simplifies experiment management and sharing, facilitating real-time discussions across different teams. TensorBoard is mainly designed for individual use.

*   **💻 Persistent Dashboard:**  Whether training locally, on a lab cluster, or in the cloud, the results are recorded in a single, centralized dashboard. TensorBoard requires you to manually copy and manage TFEvent files.

*   **💪 Enhanced Tables:**  SwanLab's tables allow you to search, filter, and compare results from different experiments easily. This makes it easier to view and find the best-performing models. TensorBoard is less suited for large projects.

### Weights and Biases vs SwanLab

*   Weights and Biases is a closed-source MLOps platform that requires an internet connection.
*   SwanLab supports both online and open-source, free, self-hosted versions.

<br>

## 👥 Community

### Related Repositories

*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation.
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard.
*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosting scripts.

### Community and Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): For bugs and questions.
*   [Email Support](zeyi.lin@swanhub.co): For feedback and support.
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Discuss SwanLab and AI tech.

### SwanLab README Badges

Add the SwanLab badge to your README:

[![][tracking-swanlab-shield]][tracking-swanlab-shield-link]
[![][visualize-swanlab-shield]][visualize-swanlab-shield-link]

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab in Your Publications

If SwanLab has been helpful in your research, please cite it:

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

Want to contribute to SwanLab? See our [Contributing Guide](CONTRIBUTING.md).

We appreciate sharing SwanLab through social media and at events!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

## 📃 License

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