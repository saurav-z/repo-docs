<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: The Open-Source AI Training Tracker and Visualizer

**SwanLab is an open-source, modern tool designed to streamline your deep learning workflows, offering comprehensive experiment tracking, visualization, and collaboration features.  Get started now at [https://github.com/SwanHubX/SwanLab](https://github.com/SwanHubX/SwanLab)!**

<br/>

## Key Features

*   **Experiment Tracking & Visualization**: Track metrics, hyperparameters, and visualize training progress with an intuitive UI.
*   **Flexible & Extensible**: Support for 30+ popular frameworks including PyTorch, 🤗HuggingFace Transformers, and many more, with plugin support for customization.
*   **Hardware Monitoring**: Monitor system-level hardware metrics (CPU, GPU, memory, etc.) in real-time.
*   **Collaboration**:  Share experiments and collaborate with your team for efficient training workflows.
*   **Self-Hosted Option**: Use SwanLab offline with the community version and still be able to manage experiments.
*   **Comprehensive Integrations**: Seamlessly integrate with popular frameworks like PyTorch, Hugging Face Transformers, and more.
*   **Advanced Charting**:  Create and customize a variety of charts including line charts, media charts (images, audio, video), 3D point clouds, and custom Echarts.

<br/>

## What's New

*   **Training Collaboration**:  Invite collaborators, share project links, and view projects in a list view with tag support.
*   **Enhanced Charting**: Flexible line chart customization, support for GIF files via `swanlab.Video`, and global chart dashboard configuration.
*   **Text View**:  Support for Markdown rendering and directional key navigation within the text view.
*   **Resume Training**:  Support for resuming training from a breakpoint.
*   **Framework Integrations**: Integrate with accelerate, AREAL, Ray and ROLL frameworks
*   **Echarts Customization**: Support for data types, group interaction, max/min metrics display.
*   **Sync**: Sync local log files with swanlab cloud.

<details><summary>See Full Changelog</summary>

*   **2025.08.06:** 👥**Training Collaboration** Released!
*   **2025.07.29:** 🚀 Sidebar support experiment filtering and sorting, table view with column control, support for multiple API keys, new charts: PR curve, ROC curve, and confusion matrix.
*   **2025.07.17:** 📊 Enhanced line chart configuration, `swanlab.Video` data type support for GIF files.
*   **2025.07.10:** 📚 Enhanced text view with Markdown rendering and arrow key navigation via `swanlab.echarts.table` and `swanlab.Text`.
*   **2025.07.06:** 🚄 Resume breakpoint training support; new file logger plugin; integrated with [ray](https://github.com/ray-project/ray) and [ROLL](https://github.com/volcengine/ROLL) frameworks.
*   **2025.06.27:** 📊 Small Line Chart Zoom, smoothing for a single line chart, and improved image chart interaction.
*   **2025.06.20:** 🤗 Integrated with [accelerate](https://github.com/huggingface/accelerate), [PR](https://github.com/huggingface/accelerate/pull/3605)
*   **2025.06.18:** 🐜 Integrated with [AREAL](https://github.com/inclusionAI/AReaL), [PR](https://github.com/inclusionAI/AReaL/pull/98) and cross-group line chart comparison, experiment name customization.
*   **2025.06.11:** 📊 Support for `swanlab.echarts.table` data type, interactive grouping, table view options for min/max.
*   **2025.06.08:** ♻️ Local experiment log file storage, hardware monitoring support for Hygon DCU.
*   **2025.06.01:** 🏸 Free chart dragging, ECharts custom chart support, hardware monitoring support for Muxi GPU.
*   **2025.05.25:** Standard error stream logging, hardware monitoring for Moore Threads.
*   **2025.05.14:** Experiment Tag Support, Log Scale for line charts, Dragging group support.
*   **2025.05.09:** Line chart creation support, chart configuration with data source selection, GitHub Badge for Training Project Generation.
*   **2025.04.23:** Line chart editing, X/Y axis range and style configuration support, Regex support for chart search.
*   **2025.04.11:** Partial region selection on line charts, global step range for the dashboard, one-click hide all charts.
*   **2025.04.08:** Molecule data type support, sort/filter/column order state persistence.
*   **2025.04.07:** Joint integration with [EvalScope](https://github.com/ModelScope/EvalScope)
*   **2025.03.30:** swanlab.Settings method support, support for Cambricon MLU hardware monitoring, Slack and Discord notification support.
*   **2025.03.21:** Hugging Face Transformers integration, Object3D chart support, GPU memory (MB), disk usage, and network metrics monitoring.
*   **2025.03.12:** SwanLab Self-Hosted version release!! [🔗 Deployment Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab now supports plugin extensions, such as [email notification](https://docs.swanlab.cn/plugin/notification-email.html) and [Feishu notification](https://docs.swanlab.cn/plugin/notification-lark.html)
*   **2025.03.09:** Sidebar widening, Git code display button, sync_mlflow function.
*   **2025.03.06:** Joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio)
*   **2025.03.04:** MLFlow Conversion Support, [Usage Guide](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html)
*   **2025.03.01:** Move Experiment Function, experimental migration between different organizations.
*   **2025.02.24:** Joint integration with [EasyR1](https://github.com/hiyouga/EasyR1)
*   **2025.02.18:** Joint integration with [Swift](https://github.com/modelscope/ms-swift)
*   **2025.02.16:** Grouping Chart Functions.
*   **2025.02.09:** Joint integration with [veRL](https://github.com/volcengine/verl)
*   **2025.02.05:** swanlab.log supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), name and notes parameters.
*   **2025.01.22:** `sync_tensorboardX` and `sync_tensorboard_torch` support.
*   **2025.01.17:** `sync_wandb` function, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html).
*   **2025.01.11:** Cloud version optimization for project table performance, drag/sort/filter interactions.
*   **2025.01.01:** Persistent smoothing, and draggable change in size and optimize chart browsing experience for line chart.
*   **2024.12.22:** Joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
*   **2024.12.15:** Hardware Monitoring (0.4.0) release with CPU, NPU (Ascend), and GPU (Nvidia) system information logging and monitoring.
*   **2024.12.06:** Integration with LightGBM and XGBoost and log limit extension
*   **2024.11.26:** Hardware tab support for Ascend NPU and Kunpeng CPU.  Cloud provider support for Qingyun Jishizhisuan.
</details>

<br/>

## Quick Start

1.  **Installation**:

```bash
pip install swanlab
```

2.  **Login**:

```bash
swanlab login
```

  Follow the prompts to enter your API key (find it at [https://swanlab.cn/settings](https://swanlab.cn/settings) after signing up).

3.  **Integrate with your code**:

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

Visit [SwanLab](https://swanlab.cn) to view your experiment.

<br/>

## Self-Hosting

Self-hosting supports offline viewing of the SwanLab dashboard.

**Deployment**:

1.  Deploy a self-hosted community version using Docker, following the instructions in the [documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

2.  Login to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

  After logging in, you can record experiments to the self-hosted service.

<br>

## Examples & Tutorials

*   [MNIST Hand-Written Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [Cifar10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)
*   [Resnet 猫狗分类](https://docs.swanlab.cn/examples/cats_dogs_classification.html)
*   [Yolo 目标检测](https://docs.swanlab.cn/examples/yolo.html)
*   [UNet医学影像分割](https://docs.swanlab.cn/examples/unet-medical-segmentation.html)
*   [音频分类](https://docs.swanlab.cn/examples/audio_classification.html)
*   [DQN强化学习-推车倒立摆](https://docs.swanlab.cn/examples/dqn_cartpole.html)
*   [LSTM Google 股票预测](https://docs.swanlab.cn/examples/audio_classification.html)
*   [BERT文本分类](https://docs.swanlab.cn/examples/bert.html)
*   [Stable Diffusion文生图微调](https://docs.swanlab.cn/examples/stable_diffusion.html)
*   [LLM预训练](https://docs.swanlab.cn/examples/pretrain_llm.html)
*   [GLM4指令微调](https://docs.swanlab.cn/examples/glm4-instruct.html)
*   [Qwen下游任务训练](https://docs.swanlab.cn/examples/qwen_finetune.html)
*   [NER命名实体识别](https://docs.swanlab.cn/examples/ner.html)
*   [Qwen3医学模型微调](https://docs.swanlab.cn/examples/qwen3-medical.html)
*   [Qwen2-VL多模态大模型微调实战](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO大模型强化学习](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B多模态模型训练](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot 具身智能入门](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)

<br/>

## Hardware Monitoring

SwanLab records **hardware information** and **resource usage** during AI training.

| Hardware        | Information Recording | Resource Monitoring | Script                                                           |
| :-------------- | :-------------------: | :-----------------: | :--------------------------------------------------------------- |
| NVIDIA GPU      |          ✅          |         ✅         | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU      |          ✅          |         ✅         | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC       |          ✅          |         ✅         | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU   |          ✅          |         ✅         | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU   |          ✅          |         ✅         | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU|         ✅         |         ✅         | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU       |         ✅         |         ✅         | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU       |          ✅          |         ✅         | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU             |          ✅          |         ✅         | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)     |
| Memory          |          ✅          |         ✅         | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk            |          ✅          |         ✅         | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)     |
| Network         |          ✅          |         ✅         | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

<br/>

## Framework Integrations

Easily integrate SwanLab with your favorite frameworks!

**Base Frameworks**

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
*   [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Finetuning Frameworks**

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

[More Integrations](https://docs.swanlab.cn/guide_cloud/integration/)

<br/>

## Plugins & API

Extend SwanLab's functionality with plugins to enhance your experiment management.

*   [Create Your Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notification](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notification](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [Dingtalk Notification](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [WeCom Notification](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notification](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notification](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Logger](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

OpenAPI
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br/>

## Comparisons

### Tensorboard vs SwanLab

-   **☁️ Online Use**: SwanLab facilitates cloud-based experiment tracking, remote access, and collaboration, while Tensorboard is a local tool.
-   **👥 Collaboration**: SwanLab supports multi-user project management, and easy sharing, Tensorboard is primarily for individuals.
-   **💻 Centralized Dashboard**: Regardless of where you train, SwanLab organizes all results into one dashboard, while TensorBoard requires managing TFEvent files.
-   **💪 Enhanced Table**: SwanLab tables enable result viewing, searching, and filtering across experiments.

### Weights and Biases vs SwanLab

-   Weights and Biases is a closed-source, and cloud-based platform.
-   SwanLab is open-source, free, and supports self-hosting.

<br/>

## Community

### Resources
-   [SwanLab-Docs](https://github.com/swanhubx/swanlab-docs): Official Documentation
-   [SwanLab-Dashboard](https://github.com/swanhubx/swanlab-dashboard): Offline Dashboard
-   [self-hosted](https://github.com/swanhubx/self-hosted): Self-hosted Script Repository

### Support
-   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues)
-   [Email Support](zeyi.lin@swanhub.co)
-   <a href="https://docs.swanlab.cn/guide_cloud/community/online-support.html">WeChat Group</a>

### SwanLab README Badges

Add these badges to your README:

[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![Visualize with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

More design assets: [assets](https://github.com/SwanHubX/assets)

### Citing SwanLab

If SwanLab has helped your research, please cite us using the following format:

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

Interested in contributing?  Read the [CONTRIBUTING.md](CONTRIBUTING.md) guide.

Share SwanLab on social media! We are very grateful for your support.

<br/>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br/>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

<br/>

## License

SwanLab is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)