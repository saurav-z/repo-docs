<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br>

## 🚀 SwanLab: The Open-Source Deep Learning Experiment Tracking & Visualization Tool

**SwanLab** is a powerful, open-source tool designed to track, visualize, and collaborate on your deep learning experiments, offering both cloud and offline capabilities and seamless integration with 30+ popular frameworks.  Explore the original repository at [https://github.com/SwanHubX/SwanLab](https://github.com/SwanHubX/SwanLab).

<br>

**Key Features:**

*   **📊 Experiment Tracking & Visualization:** Monitor metrics, visualize training progress, and analyze results with intuitive charts and dashboards.
*   **⚡️ Framework Integrations:**  Seamlessly integrates with popular ML frameworks like PyTorch, Hugging Face Transformers, and many more (see below).
*   **💻 Hardware Monitoring:** Real-time tracking of CPU, GPU (Nvidia, Ascend, etc.), and memory usage for comprehensive system insights.
*   **📦 Experiment Management:**  Organize and manage experiments with a centralized dashboard for easy comparison and analysis.
*   **🆚 Result Comparison:**  Compare and contrast experiment results to identify trends and optimize model performance.
*   **👥 Collaboration:**  Enable team collaboration by sharing experiments, insights, and ideas.
*   **☁️ Cloud & Offline Support:**  Use SwanLab online or self-host for complete control and flexibility.
*   **🔌 Plugin Ecosystem:** Extend SwanLab's functionality with custom plugins and integrations (e.g., Slack, Discord, CSV writers).

**Why Use SwanLab?**

- **Simplify Your Workflow:** Reduce the complexity of tracking, visualizing, and sharing your deep learning experiments.
- **Accelerate Research:** Gain insights faster and iterate more effectively by visualizing your training data.
- **Improve Collaboration:** Streamline team communication and knowledge sharing with collaborative features.
- **Enhance Reproducibility:**  Easily reproduce experiments and share results with colleagues.

**Core Functionalities:**

1.  **Comprehensive Tracking**: Log metrics, hyperparameter, system environment, hardware info, etc.
2.  **Versatile Charting**: Visualize scalar metrics, images, videos, and custom charts.
3.  **Framework Compatibility**: Integrate with popular frameworks.
4.  **Rich Integrations**:  Connect with various platforms for extended capabilities (e.g., cloud storage, notifications).

**See SwanLab in Action!**

**Online Demo:**

*   [ResNet50 猫狗分类][demo-cats-dogs] | [Yolov8-COCO128 目标检测][demo-yolo]
*   [Qwen2 指令微调][demo-qwen2-sft] | [LSTM Google 股票预测][demo-google-stock]
*   [ResNeXt101 音频分类][demo-audio-classification] | [Qwen2-VL COCO数据集微调][demo-qwen2-vl]
*   [EasyR1 多模态LLM RL训练][demo-easyr1-rl] | [Qwen2.5-0.5B GRPO训练][demo-qwen2-grpo]

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## 🚀 Getting Started Quickly

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

### 2. Log in and Get Your API Key

1.  [Register](https://swanlab.cn) for a free account.

2.  Log in and copy your API key from User Settings > [API Key](https://swanlab.cn/settings).

3.  Run in your terminal:

```bash
swanlab login
```

Enter your API key when prompted.

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

Access your experiment from [SwanLab](https://swanlab.cn).

<br>

## 💻 Self-Hosting

SwanLab can be self-hosted for offline use.  Deploy SwanLab with Docker:

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

Then run the installation script.

<br>

## 🚗 Framework Integrations

Integrate SwanLab with your favorite ML frameworks!  Here's a list of currently supported frameworks:

**Base Frameworks**
- [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
- [MindSpore](https://docs.swanlab.cn/guide_cloud/integration/integration-ascend.html)
- [Keras](https://docs.swanlab.cn/guide_cloud/integration/integration-keras.html)

**Specialized/Fine-Tuning Frameworks**
- [PyTorch Lightning](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-lightning.html)
- [HuggingFace Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-transformers.html)
- [LLaMA Factory](https://docs.swanlab.cn/guide_cloud/integration/integration-llama-factory.html)
- [Modelscope Swift](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html)
- [DiffSynth Studio](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html)
- [Sentence Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-sentence-transformers.html)
- [PaddleNLP](https://docs.swanlab.cn/guide_cloud/integration/integration-paddlenlp.html)
- [OpenMind](https://modelers.cn/docs/zh/openmind-library/1.0.0/basic_tutorial/finetune/finetune_pt.html#%E8%AE%AD%E7%BB%83%E7%9B%91%E6%8E%A7)
- [Torchtune](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch-torchtune.html)
- [XTuner](https://docs.swanlab.cn/guide_cloud/integration/integration-xtuner.html)
- [MMEngine](https://docs.swanlab.cn/guide_cloud/integration/integration-mmengine.html)
- [FastAI](https://docs.swanlab.cn/guide_cloud/integration/integration-fastai.html)
- [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html)
- [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html)
- [MLX-LM](https://docs.swanlab.cn/guide_cloud/integration/integration-mlx-lm.html)

**Evaluation Frameworks**
- [EvalScope](https://docs.swanlab.cn/guide_cloud/integration/integration-evalscope.html)

**Computer Vision**
- [Ultralytics](https://docs.swanlab.cn/guide_cloud/integration/integration-ultralytics.html)
- [MMDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-mmdetection.html)
- [MMSegmentation](https://docs.swanlab.cn/guide_cloud/integration/integration-mmsegmentation.html)
- [PaddleDetection](https://docs.swanlab.cn/guide_cloud/integration/integration-paddledetection.html)
- [PaddleYOLO](https://docs.swanlab.cn/guide_cloud/integration/integration-paddleyolo.html)

**Reinforcement Learning**
- [Stable Baseline3](https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html)
- [veRL](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)
- [HuggingFace trl](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-trl.html)
- [EasyR1](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
- [AReaL](https://docs.swanlab.cn/guide_cloud/integration/integration-areal.html)
- [ROLL](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)

**Other Frameworks：**
- [Tensorboard](https://docs.swanlab.cn/guide_cloud/integration/integration-tensorboard.html)
- [Weights&Biases](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html)
- [MLFlow](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html)
- [HuggingFace Accelerate](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html)
- [Ray](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html)
- [Unsloth](https://docs.swanlab.cn/guide_cloud/integration/integration-unsloth.html)
- [Hydra](https://docs.swanlab.cn/guide_cloud/integration/integration-hydra.html)
- [Omegaconf](https://docs.swanlab.cn/guide_cloud/integration/integration-omegaconf.html)
- [OpenAI](https://docs.swanlab.cn/guide_cloud/integration/integration-openai.html)
- [ZhipuAI](https://docs.swanlab.cn/guide_cloud/integration/integration-zhipuai.html)

<br>

## 🔌 Plugins and API

Extend SwanLab's capabilities with plugins!  [See Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html).

<br>

## 👥 Community & Support

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues) - Report bugs and ask questions.
*   [Email Support](zeyi.lin@swanhub.co) - Get help with SwanLab.
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html) - Connect with other users.

<br>

## 📃 License

SwanLab is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).