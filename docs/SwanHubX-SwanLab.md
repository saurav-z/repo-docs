<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

# SwanLab: Revolutionizing Deep Learning Experiment Tracking and Visualization

**SwanLab is an open-source, modern, and user-friendly tool for tracking, visualizing, and collaborating on your deep learning experiments, streamlining your workflow and accelerating your research.  [See the original repo](https://github.com/SwanHubX/SwanLab)**

Key features and benefits:

*   **Intuitive Experiment Tracking**: Easily log and visualize metrics, hyperparameters, and artifacts to understand your model's performance.
*   **Unified Dashboard**: Centralized dashboard to manage and organize your projects and experiments.
*   **Extensive Framework Integrations**: Seamlessly integrates with 30+ popular deep learning frameworks, including PyTorch, TensorFlow, and Hugging Face Transformers.
*   **Hardware Monitoring**: Real-time monitoring of CPU, GPU, memory, and other system resources.
*   **Experiment Comparison**: Compare and analyze results across different experiments with interactive charts and tables.
*   **Team Collaboration**: Facilitate collaboration with team members through shared projects and experiment results.
*   **Cloud and Offline Support**: Use SwanLab in the cloud or on your local machine.
*   **Customizable & Extendable**: Extend functionality with plugins for notifications, integrations, and more.
*   **Community Driven**: Join a vibrant community of researchers and developers.

## Table of Contents

*   [🚀 Recent Updates](#-最近更新)
*   [✨ Key Features](#-key-features)
*   [💻 Quick Start](#-quick-start)
*   [📦 Self-Hosting](#-自托管)
*   [🚀 Example Demos](#-在线演示)
*   [⚙️ Integrations](#-框架集成)
*   [🤝 Community & Support](#-社区)
*   [📄 License](#-协议)
*   [🌟 Star History](#-star-history)

<br>

## 🚀 Recent Updates

*   (Latest updates are listed here, keep it updated)

### **Recent updates are listed in the original README.  Please see it [here](https://github.com/SwanHubX/SwanLab)**
<br>

## ✨ Key Features

*   **Experiment Tracking and Visualization**: Real-time visualization of training metrics, hyperparameters, and model performance.
    *   **Supported Data Types**: Scalars, Images, Audio, Text, Videos, 3D Point Clouds, Biochemical Molecules, and ECharts Custom Charts.
    *   **Chart Types**: Line charts, Media charts, 3D point clouds, Biochemical Molecules, Bar charts, Scatter plots, Box plots, Heatmaps, Pie charts, and Radar charts.
*   **Framework Integrations**: Seamless integration with popular deep learning frameworks.
*   **Hardware Monitoring**: Monitor system resources (CPU, GPU, memory, etc.).
*   **Experiment Management**: Organize and manage experiments in a centralized dashboard.
*   **Comparison Tools**: Compare and analyze experiment results using tables and interactive charts.
*   **Collaboration Features**: Share results and collaborate with your team.
*   **Self-Hosting**: Support for local/offline use.
*   **Plugin Ecosystem**: Expand functionality with plugins.

## 💻 Quick Start

1.  **Installation**:

    ```bash
    pip install swanlab
    ```

2.  **Login (Get your API Key):**
    *   Register an account here: [https://swanlab.cn](https://swanlab.cn)
    *   Login to [SwanLab](https://swanlab.cn) and copy your API Key

    ```bash
    swanlab login
    ```
    Enter your API Key when prompted.

3.  **Integrate into your Code**:

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

## 📦 Self-Hosting

SwanLab supports self-hosting for local use or private deployments.

### Deploy with Docker

1.  Clone the repository:

    ```bash
    git clone https://github.com/SwanHubX/self-hosted.git
    cd self-hosted/docker
    ```

2.  Run the installation script:

    ```bash
    ./install.sh  # For China mainland users
    ```
    OR
    ```bash
    ./install-dockerhub.sh # From DockerHub
    ```

    Complete setup instructions are in the [official documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

3.  Log in to your self-hosted instance:

    ```bash
    swanlab login --host http://localhost:8000
    ```

## 🚀 Example Demos

Explore the online demos to see SwanLab in action:

**Each example has interactive visualization, see the original README for links**
*   ResNet50 for Cat/Dog Classification
*   Yolov8 for COCO128 Object Detection
*   Qwen2 Instruction Fine-tuning
*   LSTM for Google Stock Prediction
*   ResNeXt101 for Audio Classification
*   Qwen2-VL COCO Fine-tuning
*   EasyR1 Multi-modal LLM RL Training
*   Qwen2.5-0.5B GRPO Training

## ⚙️ Integrations

SwanLab seamlessly integrates with various popular frameworks, including:

*   PyTorch, TensorFlow, Keras, and more.
*   Hugging Face Transformers, PyTorch Lightning, LLaMA Factory,  and many others.
*   Ultralytics, MMDetection, PaddleDetection and others.
*   Stable Baseline3, veRL, HuggingFace trl, and EasyR1 for Reinforcement Learning.
*   Tensorboard, Weights & Biases, and other tools.

For more details, check out the  [complete list](https://docs.swanlab.cn/guide_cloud/integration/).

## 🤝 Community & Support

*   **GitHub Issues**: Report issues and ask questions [here](https://github.com/SwanHubX/SwanLab/issues).
*   **Email Support**: Send feedback via email (zeyi.lin@swanhub.co).
*   **WeChat Group**:  Join the WeChat group for discussions and support [here](https://docs.swanlab.cn/guide_cloud/community/online-support.html).
*   **Contribute**: Contribute to SwanLab [here](CONTRIBUTING.md).

## 📄 License

SwanLab is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).

## 🌟 Star History

```
[![Star History Chart](https://api.star-history.com/svg?repos=swanhubx/swanlab&type=Date)](https://star-history.com/#swanhubx/swanlab&Date)