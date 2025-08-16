<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
    <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
  </picture>
</div>

# SwanLab: Supercharge Your Deep Learning Experiments with Ease

**SwanLab is an open-source, user-friendly tool that revolutionizes deep learning experiment tracking and visualization, simplifying your workflow with intuitive features and robust integrations.**. Designed for researchers and practitioners, SwanLab offers a seamless way to track metrics, visualize training progress, and collaborate on projects.  

[🔥SwanLab Online](https://swanlab.cn) | [📃 Documentation](https://docs.swanlab.cn) | [🐛 Report Issues](https://github.com/swanhubx/swanlab/issues) | [💡 Feedback](https://geektechstudio.feishu.cn/share/base/form/shrcnyBlK8OMD0eweoFcc2SvWKc) | [📜 Changelog](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html) | [🤝 Community](https://swanlab.cn/benchmarks)

[![Release](https://img.shields.io/github/v/release/swanhubx/swanlab?color=369eff&labelColor=black&logo=github&style=flat-square)](https://github.com/swanhubx/swanlab/releases)
[![Docker Hub](https://img.shields.io/docker/v/swanlab/swanlab-next?color=369eff&label=docker&labelColor=black&logoColor=white&style=flat-square)](https://hub.docker.com/r/swanlab/swanlab-next/tags)
[![GitHub Stars](https://img.shields.io/github/stars/swanhubx/swanlab?labelColor&style=flat-square&color=ffcb47)](https://github.com/swanhubx/swanlab)
[![GitHub Issues](https://img.shields.io/github/issues/swanhubx/swanlab?labelColor=black&style=flat-square&color=ff80eb)](https://github.com/swanhubx/swanlab/issues)
[![Contributors](https://img.shields.io/github/contributors/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/graphs/contributors)
[![License](https://img.shields.io/badge/license-apache%202.0-white?labelColor=black&style=flat-square)](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)
[![Tracking with SwanLab](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](https://swanlab.cn)
[![Last Commit](https://img.shields.io/github/last-commit/swanhubx/swanlab?color=c4f042&labelColor=black&style=flat-square)](https://github.com/swanhubx/swanlab/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/swanlab?color=orange&labelColor=black&style=flat-square)](https://pypi.org/project/swanlab/)
[![WeChat Group](https://img.shields.io/badge/WeChat-微信-4cb55e?labelColor=black&style=flat-square)](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
[![PyPI Downloads](https://static.pepy.tech/badge/swanlab?labelColor=black&style=flat-square)](https://pepy.tech/project/swanlab)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

<br/>

[![](readme_files/swanlab-overview.png)](https://swanlab.cn)

[中文 / English](README_EN.md) / [日本語](README_JP.md) / [Русский](README_RU.md)

👋 Join our [WeChat group](https://docs.swanlab.cn/zh/guide_cloud/community/online-support.html)

<a href="https://hellogithub.com/repository/b442a9fa270e4ccb8847c9ee3445e41b" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=b442a9fa270e4ccb8847c9ee3445e41b&claim_uid=Oh5UaGjfrblg0yZ" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

<br/>

## Key Features

*   **Effortless Experiment Tracking:** Log and visualize key metrics, hyperparameters, and more with a few lines of Python code.

*   **Real-time Visualization:** Monitor training progress with interactive charts, graphs, and dashboards.

*   **Framework-Agnostic:** Seamlessly integrates with popular deep learning frameworks, including PyTorch, TensorFlow, and many others.

*   **Cloud & Self-Hosted Options:** Use SwanLab online for collaborative projects or self-host for complete data control and offline access.

*   **Advanced Data Types:** Supports various data types, including scalars, images, audio, text, video, 3D point clouds, biochemical molecules, and custom ECharts.

*   **Hardware Monitoring:** Real-time tracking of CPU, GPU (Nvidia, AMD), memory, and disk usage for performance insights.

*   **Experiment Comparison:**  Compare results, analyze hyperparameters, and identify the best-performing models.

*   **Collaboration Features:**  Share experiments with your team, leave comments, and work together to accelerate research.

*   **Flexible Sharing:** Generate shareable links for experiments to showcase results and collaborate with others.

*   **Extensible with Plugins:** Extend the functionalities of SwanLab via plugins such as [Lark Notification](https://docs.swanlab.cn/plugin/notification-lark.html) & [Slack Notification](https://docs.swanlab.cn/plugin/notification-slack.html)

<br/>

## Recent Updates

*   **[2025.08.06]:** 👥**Training Collaboration** launched, enabling the invitation of project collaborators, sharing project links and QR codes; workspace supports list view, displaying project tags;
*   **[2025.07.29]:** 🚀 Sidebars support **experiment filtering, sorting**; 📊 Table view goes online with **column control panel**, making it easy to hide and show columns; 🔐 **Multi-API Key** management goes live to enhance your data security; swanlab sync improves compatibility for log file integrity, adapting to scenarios like training crashes; New charts - PR curve, ROC curve, confusion matrix go live, [Documentation](https://docs.swanlab.cn/api/py-pr_curve.html);
*   **[2025.07.17]:** 📊 More powerful **line chart configuration**, supporting flexible configuration of line types, colors, thicknesses, grids, and legend positions; 📹 Support **swanlab.Video** data type, allowing recording and visualization of GIF format files; Global chart dashboard supports configuring the Y-axis and the maximum number of experiments to display;
*   **[2025.07.10]:** 📚 More powerful **text view**, supporting Markdown rendering and arrow key switching, created by `swanlab.echarts.table` and `swanlab.Text`, [Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)
*   **[2025.07.06]:** 🚄 Support **resume breakpoint training**; new plugin **file recorder**; integration with [ray](https://github.com/ray-project/ray) framework, [Documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); integration with [ROLL](https://github.com/volcengine/ROLL) framework, thanks to [@PanAndy](https://github.com/PanAndy), [Documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)
*   **[2025.06.27]:** 📊 Support **small line chart local zoom**; support configuring **single line chart smoothing**; significantly improved the interaction effect after image chart zoom;
*   **[2025.06.20]:** 🤗 Integrates the [accelerate](https://github.com/huggingface/accelerate) framework, [PR](https://github.com/huggingface/accelerate/pull/3605), [Documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html), enhancing the experiment logging experience in distributed training;
*   **[2025.06.18]:** 🐜 Integrated the [AREAL](https://github.com/inclusionAI/AReaL) framework, thanks to [@xichengpro](https://github.com/xichengpro), [PR](https://github.com/inclusionAI/AReaL/pull/98), [Documentation](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process); 🖱 Support mouse hover to the sidebar experiment, highlighting the corresponding curve; Support cross-group comparison of line charts; Support setting experiment name cropping rules;
*   **[2025.06.11]:** 📊 Support **swanlab.echarts.table** data type, support pure text chart display; Support **stretching interaction** of grouping to increase the number of charts displayed simultaneously; Table view increases **index maximum/minimum** option;

<details><summary>More Updates</summary>

-   [2025.06.08]: ♻️ Support for storing complete experiment log files locally, uploading local log files to the cloud/privatized deployment via **swanlab sync**; Hardware monitoring supports **Hygon DCU**;
-   [2025.06.01]: 🏸 Support **chart free drag and drop**; Support **ECharts custom chart**, adding more than 20 chart types including bar chart, pie chart, histogram; Hardware monitoring supports **MetaX GPU**; Integrate **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** framework;
-   [2025.05.25]: Logs support recording **standard error stream**, printing information from frameworks like PyTorch Lightning can be better recorded; Hardware monitoring supports **Moore Threads**; Added the running command recording security protection function, API Key will be automatically hidden;
-   [2025.05.14]: Support **experiment Tag**; Support line chart **Log Scale**; Support **grouping drag and drop**; Greatly optimize the experience of uploading a large number of metrics; Added `swanlab.OpenApi` open interface;
-   [2025.05.09]: Support **line chart creation**; Configuration chart function adds **data source selection** function, supports displaying different indicators on a single chart; Support generating **training project GitHub badge**;
-   [2025.04.23]: Support line chart **editing**, support freely configuring the X and Y axis data range and title style of the chart; Chart search supports **regular expressions**; Support hardware detection and monitoring of **Kunlunxin XPU**;
-   [2025.04.11]: Support line chart **local area selection**; Support global selection of the step range of the dashboard line chart; Support one-click hide all charts;
-   [2025.04.08]: Support **swanlab.Molecule** data type, support recording and visualizing biochemical molecule data; Support saving the sorting, filtering, and column order change states in the table view;
-   [2025.04.07]: We have completed a joint integration with [EvalScope](https://github.com/ModelScope/EvalScope), now you can use SwanLab in EvalScope to **evaluate the performance of large models**;
-   [2025.03.30]: Support the **swanlab.Settings** method, support more refined experimental behavior control; Support **Cambricon MLU** hardware monitoring; Support [Slack notification](https://docs.swanlab.cn/plugin/notification-slack.html), [Discord notification](https://docs.swanlab.cn/plugin/notification-discord.html);
-   [2025.03.21]: 🎉🤗HuggingFace Transformers has officially integrated SwanLab (>=4.50.0 version), [#36433](https://github.com/huggingface/transformers/pull/36433); New **Object3D chart**, support recording and visualizing 3D point cloud, [Documentation](https://docs.swanlab.cn/api/py-object3d.html); Hardware monitoring supports the recording of GPU video memory (MB), disk usage, and network up and down;
-   [2025.03.12]: 🎉🎉SwanLab **private deployment version** has been released! [🔗Deployment documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab now supports plugin extensions, such as [email notification](https://docs.swanlab.cn/plugin/notification-email.html), [Lark notification](https://docs.swanlab.cn/plugin/notification-lark.html)
-   [2025.03.09]: Support **experiment sidebar widening**; Added an external display Git code button; Added the **sync\_mlflow** function, supporting the synchronization of experimental tracking with the mlflow framework;
-   [2025.03.06]: We have completed a joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio), now you can use SwanLab in DiffSynth Studio to **track and visualize Diffusion model text-to-image/video experiments**, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html);
-   [2025.03.04]: Added the **MLFlow conversion** function, supporting the conversion of MLFlow experiments to SwanLab experiments, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html);
-   [2025.03.01]: Added the **move experiment** function, now you can move experiments to different projects of different organizations;
-   [2025.02.24]: We have completed a joint integration with [EasyR1](https://github.com/hiyouga/EasyR1), now you can use SwanLab in EasyR1 to **track and visualize multimodal large model reinforcement learning experiments**, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
-   [2025.02.18]: We have completed a joint integration with [Swift](https://github.com/modelscope/ms-swift), now you can use SwanLab in Swift's CLI/WebUI to **track and visualize large model fine-tuning experiments**, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html).
-   [2025.02.16]: Added the **chart move grouping, create grouping** function.
-   [2025.02.09]: We have completed a joint integration with [veRL](https://github.com/volcengine/verl), now you can use SwanLab in veRL to **track and visualize large model reinforcement learning experiments**, [usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html).
-   [2025.02.05]: `swanlab.log` supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), adapting to Jax framework characteristics; supports `name` and `notes` parameters;
-   [2025.01.22]: Added `sync_tensorboardX` and `sync_tensorboard_torch` functions, supporting experimental tracking synchronization with these two TensorBoard frameworks;
-   [2025.01.17]: Added the `sync_wandb` function, [Documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html), supporting synchronization with Weights & Biases experimental tracking; Greatly improved log rendering performance
-   [2025.01.11]: The cloud version greatly optimizes the performance of the project table and supports interactions such as drag and drop, sorting, and filtering
-   [2025.01.01]: Added line chart **persistent smoothing**, drag and drop to change the size of the line chart, optimizing the chart browsing experience
-   [2024.12.22]: We have completed a joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), now you can use SwanLab in LLaMA Factory to **track and visualize large model fine-tuning experiments**, [usage guide](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger).
-   [2024.12.15]: **Hardware monitoring (0.4.0)** function goes live, supporting CPU, NPU (Ascend), GPU (Nvidia) system-level information recording and monitoring.
-   [2024.12.06]: Added the integration of [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html), [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html); Increased the limit on the length of a single line of log recording.
-   [2024.11.26]: Environment tab - The hardware section supports identifying **Huawei Ascend NPU** and **Kunpeng CPU**; The cloud vendor section supports identifying Qingyun **Jishi Zhisuan**.
</details>

<br/>

## Quickstart

### 1. Installation

```bash
pip install swanlab
```

<details><summary>Source Install</summary>

```bash
# Method 1
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .

# Method 2
pip install git+https://github.com/SwanHubX/SwanLab.git
```

</details>

<details><summary>Offline Dashboard Install</summary>

```bash
pip install 'swanlab[dashboard]'
```

</details>

### 2. Login and Get Your API Key

1.  [Register a free account](https://swanlab.cn)
2.  Log in to your account and copy your API Key from User Settings > [API Key](https://swanlab.cn/settings).
3.  Open your terminal and enter:

```bash
swanlab login
```

Enter your API Key when prompted and press Enter to log in.

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

Congratulations!  Go to [SwanLab](https://swanlab.cn) to see your first experiment!

<br/>

## Self-Hosting

Self-hosting the Community Edition allows you to view the SwanLab dashboard offline.

![swanlab-docker](./readme_files/swanlab-docker.png)

### 1. Deploy Self-Hosted Version Using Docker

For details, please refer to: [Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

For China region fast installation:

```bash
./install.sh
```

Install from DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Point Experiments to Your Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

After logging in, you can record experiments to your self-hosted service.

<br/>

## Real-World Examples

**Open-source projects using SwanLab:**

*   [happy-llm](https://github.com/datawhalechina/happy-llm): A tutorial on the principles and practice of large language models from scratch. ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/happy-llm)
*   [self-llm](https://github.com/datawhalechina/self-llm): "Open Source Large Model Food Guide" customized for Chinese users, based on Linux environment for rapid fine-tuning (full parameter/Lora), deploying domestic and foreign open source large models (LLM)/multimodal large models (MLLM) tutorial ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/self-llm)
*   [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek): DeepSeek series work interpretation, extension and reproduction ![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/unlock-deepseek)
*   [Qwen3-SmVL](https://github.com/ShaohonChen/Qwen3-SmVL): The visual head of SmolVLM2 is spliced and fine-tuned with the Qwen3-0.6B model ![GitHub Repo stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL)

**Papers using SwanLab:**

*   [Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models](https://arxiv.org/abs/2507.03916)
*   [Efficient Model Fine-Tuning with LoRA for Biomedical Named Entity Recognition](https://ieeexplore.ieee.org/abstract/document/11082049/)
*   [SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy](https://arxiv.org/abs/2508.01188)
*   [CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL](https://arxiv.org/pdf/2508.05242)

**Tutorials:**

*   [MNIST Handwritten Digit Recognition](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothing Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
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
*   [Qwen2-VL Multimodal Large Model Fine-tuning Practice](https://docs.swanlab.cn/examples/qwen_vl_coco.html)
*   [GRPO Large Model Reinforcement Learning](https://docs.swanlab.cn/examples/qwen_grpo.html)
*   [Qwen3-SmVL-0.6B Multimodal Model Training](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
*   [LeRobot Embodied Intelligence Introduction](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)
*   [GLM-4.5-Air-LoRA and SwanLab Visualized Recording](https://github.com/datawhalechina/self-llm/blob/master/models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20%E5%8F%8A%20Swanlab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%BE%AE%E8%B0%83.md)

🌟If you have a tutorial you want to include, please submit a PR!

<br/>

## Hardware Monitoring

SwanLab records **hardware information** and **resource usage** during AI training. Here's the support table:

| Hardware           | Information Recorded | Resource Monitoring | Script                                                                   |
| ------------------ | -------------------- | ------------------- | ------------------------------------------------------------------------ |
| NVIDIA GPU         | ✅                   | ✅                  | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)   |
| Ascend NPU         | ✅                   | ✅                  | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)   |
| Apple SOC          | ✅                   | ✅                  | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)    |
| Cambricon MLU      | ✅                   | ✅                  | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU      | ✅                   | ✅                  | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py)
| Metax GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py)
| Hygon DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py)
| CPU                | ✅                   | ✅                  | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)         |
| Memory             | ✅                   | ✅                  | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)      |
| Disk               | ✅                   | ✅                  | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)        |
| Network            | ✅                   | ✅                  | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)     |

If you want to record other hardware, welcome to submit an Issue and PR!

<br/>

## Framework Integrations

Use SwanLab with your favorite framework!  
Here's a list of the frameworks we've integrated, and we welcome you to submit an [Issue](https://github.com/swanhubx/swanlab/issues) to request your desired framework.

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
*   [ZhipuAI](https://docs.swanlab.cn/guide_cloud/integration/