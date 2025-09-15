<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: Your Open-Source Toolkit for Deep Learning Experiment Tracking and Visualization

**SwanLab empowers AI researchers to visualize, track, compare, and collaborate on machine learning experiments with ease.** It offers a user-friendly Python API, a beautiful UI, and seamless integration with 30+ popular frameworks. [Explore the SwanLab Repository](https://github.com/SwanHubX/SwanLab).

[🔥SwanLab Online Version](https://swanlab.cn)  ·  [📃 Documentation](https://docs.swanlab.cn)  ·  [Report Issues](https://github.com/swanhubx/swanlab/issues) ·  [Suggestions/Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) · [Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html)  ·  <img height="16" width="16" src="https://raw.githubusercontent.com/SwanHubX/assets/main/community.svg" alt="swanlab community Logo" /> [Benchmark Community](https://swanlab.cn/benchmarks)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![DockerHub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

![](readme_files/swanlab-overview.png)

中文 / [English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat Group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features of SwanLab

*   **Experiment Tracking and Visualization**:

    *   **Cloud-based and Offline Support**: Use SwanLab seamlessly in the cloud (similar to Weights & Biases) or self-host for complete control.
    *   **Comprehensive Metadata Tracking**: Log hyperparameters, metrics, and table analysis.
    *   **Rich Visualization**: Visualize training progress with intuitive UI:
        *   Supported Metadata Types: Scalar metrics, images, audio, text, videos, 3D point clouds, biochemical molecules, custom ECharts charts.
        *   Supported Chart Types: Line charts, media charts (images, audio, text, videos), 3D point clouds, biochemical molecules, bar charts, scatter plots, box plots, heatmaps, pie charts, radar charts, custom charts.
        *   LLM-Generated Content Visualization: Text content visualization charts tailored for large language model training scenarios, supporting Markdown rendering.

        ![swanlab-table](readme_files/molecule.gif)
        ![swanlab-echarts](readme_files/echarts.png)
        ![text-chart](readme_files/text-chart.gif)
    *   **Automatic Logging**: Log system information, hardware environment, Git repository, Python environment, and project directory.
    *   **Resume Training**: Supports resuming training after completion/interruption to add new metric data to the same experiment.
*   **Framework Integrations**: Integrates with over 30 popular frameworks, including:

    *   PyTorch, Hugging Face Transformers, PyTorch Lightning, LLaMA Factory, MMDetection, Ultralytics, PaddleDetetion, LightGBM, XGBoost, Keras, Tensorboard, Weights&Biases, OpenAI, Swift, XTuner, Stable Baseline3, Hydra.

    ![](readme_files/integrations.png)
*   **Hardware Monitoring**: Real-time monitoring of hardware metrics, including CPU, NPU (Ascend), GPU (Nvidia), MLU (Cambricon), XPU (Kunlunxin), DCU (Hygon), MetaX GPU (Muxi), Moore Threads GPU, memory.
*   **Experiment Management**: A centralized dashboard designed for training scenarios to quickly manage multiple projects and experiments.
*   **Comparison of Results**: Compare hyperparameters and results from different experiments with online tables and comparison charts, discover innovation opportunities.
    ![swanlab-table](readme_files/swanlab-table.png)
*   **Online Collaboration**: Support collaborative training with your team by sharing experimental links in a project, you can check the team's training records, based on the results of the views and suggestions
*   **Result Sharing**: Share experiment results by copying and sending persistent URLs to share experiments with partners or embed them in online notes.
*   **Self-Hosting**: Supports offline use. The self-hosted community version can also view the dashboard and manage experiments. See [Self-Hosting](#-自托管) for details.
*   **Plugin Extensibility**: Extends SwanLab's application scenarios through plugins, such as [Feishu notification](https://docs.swanlab.cn/plugin/notification-lark.html), [Slack notification](https://docs.swanlab.cn/plugin/notification-slack.html), and [CSV recorder](https://docs.swanlab.cn/plugin/writer-csv.html).

> \[!IMPORTANT]
>
> **Star the project** to receive all release notifications without delay! ⭐️

![star-us](readme_files/star-us.png)

<br>

## Quick Start

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Install from Source</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
cd SwanLab
pip install -e .
```

</details>

### 2. Login and Get API Key

1.  [Register for a free account](https://swanlab.cn)
2.  Log in to your account, and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings)
3.  Open a terminal and enter:

```bash
swanlab login
```

Enter your API Key when prompted, and press Enter to complete the login.

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

Congratulations! Now, view your first SwanLab experiment at [SwanLab](https://swanlab.cn).

<br>

## Self-Hosting

The self-hosted community version supports offline viewing of the SwanLab dashboard.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted Version Using Docker

For details, refer to: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Fast installation in China:

```bash
./install.sh
```

Installation by pulling images from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Specify Experiments to Self-Hosted Service

Log in to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, you can log experiments to the self-hosted service.

<br>

## Real-World Examples

- [Happy-LLM](https://github.com/datawhalechina/happy-llm): A Tutorial for Deep Learning with the Language Model
- [Self-LLM](https://github.com/datawhalechina/self-llm):  The "Open Source Large Model User Guide" tailored for Chinese users, a tutorial based on Linux environment for quickly fine-tuning (full parameter/Lora), deploying domestic and foreign open source large models (LLM)/multimodal large models (MLLM)
- [Unlock-Deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, expansion and replication
- [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL):  The visual head of SmolVLM2 is spliced and fine-tuned with the Qwen3-0.6B model
- [OPPO/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models): End-to-end Agent foundation models through multi-Agent distillation and Agent RL

**Papers Using SwanLab:**

- [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
- [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
- [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
- [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorials:**

*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothes Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet Cat and Dog Classification](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo Object Detection](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet Medical Image Segmentation](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [Audio Classification](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN Reinforcement Learning - Cartpole](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google Stock Prediction](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT Text Classification](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion Text-to-Image Fine-tuning](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM Pre-training](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4 Instruction Fine-tuning](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen Downstream Task Training](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER Named Entity Recognition](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3 Medical Model Fine-tuning](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL Multimodal Large Model Fine-tuning](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence Getting Started](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA and SwanLab Visualization Record](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)
*   [How to Do RAG? The SwanLab Document Assistant Program is Open Source](https://docs.swanlab.cn/course/prompt_engineering_course/11-swanlab_rag/1.swanlab-rag.html)

🌟 Welcome to submit PRs if you have tutorials that you want to include!

<br>

## Hardware Monitoring

SwanLab records the hardware information and resource usage during the AI training process, the following is the support status table:

| Hardware | Information Recording | Resource Monitoring | Script |
| --- | --- | --- | --- |
| NVIDIA GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU | ✅ | ✅ | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC | ✅ | ✅ | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU | ✅ | ✅ | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU | ✅ | ✅ | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU     | ✅        | ✅        | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory        | ✅        | ✅        | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk        | ✅        | ✅        | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

If you want to record other hardware, welcome to submit Issue and PR!

<br>

## Framework Integrations

Integrate your favorite frameworks with SwanLab!  
Here is a list of frameworks we have integrated, welcome to submit [Issue](https://github.com/swanhubx/swanlab/issues) to give feedback on frameworks you want to integrate.

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
*   [MLX-LM](https://docs.swanlab.cn/guide_cloud/integration/integration-mlx-lm.html)

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

**Other Frameworks：**
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

## Plugins and API

Extend SwanLab's functionality with plugins and enhance your experiment management experience!

*   [Custom your plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notification](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notification](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notification](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeChat Work Notification](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notification](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notification](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Recorder](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Logger](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open APIs:
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparison with Familiar Tools

### Tensorboard vs SwanLab

*   **☁️ Online Support**:
    SwanLab provides convenient online synchronization and saving of training experiments in the cloud, making it easier to view training progress remotely, manage historical projects, share experiment links, send real-time message notifications, and view experiments on multiple devices. Tensorboard is an offline experiment tracking tool.

*   **👥 Collaboration**:
    When conducting machine learning collaborations among multiple people or across teams, SwanLab makes it easy to manage training projects, share experiment links, and exchange discussions across spaces. Tensorboard is mainly designed for individual use and is difficult to use for collaboration and sharing of experiments.

*   **💻 Persistent, Centralized Dashboard**:
    No matter where you train your model, whether on a local computer, in a lab cluster, or in a public cloud GPU instance, your results will be recorded on the same centralized dashboard. TensorBoard requires time to copy and manage TFEvent files from different machines.

*   **💪 More Powerful Tables**:
    SwanLab tables allow you to view, search, and filter results from different experiments, making it easy to view thousands of model versions and find the best performing model for different tasks. TensorBoard is not suitable for large projects.

### Weights and Biases vs SwanLab

*   Weights and Biases is a closed-source MLOps platform that requires internet connectivity.

*   SwanLab not only supports internet connectivity but also provides open source, free, and self-hosted versions.

<br>

## Community

### Related Repositories

*   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosted deployment script repository
*   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official documentation repository
*   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline dashboard repository, which stores the web code of the lightweight offline dashboard opened by `swanlab watch`

### Community and Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): Errors and problems encountered when using SwanLab
*   [Email Support](zeyi.lin@swanhub.co): Feedback on questions about using SwanLab
*   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>: Discussing issues using SwanLab, sharing the latest AI technology

### SwanLab README Badges

If you like using SwanLab in your work, add the SwanLab badge to your README:

[![SwanLab Tracking](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![SwanLab Visualizing](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Cite SwanLab in your Paper

If you find SwanLab helpful for your research journey, please consider citing in the following format:

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

Interested in contributing to SwanLab? Please take a moment to read the [Contribution Guidelines](CONTRIBUTING.md) first.

We also welcome support for SwanLab through sharing on social media, events, and conferences, and we sincerely appreciate it!

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

<br/>

## License

This repository is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)