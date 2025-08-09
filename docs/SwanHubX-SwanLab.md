<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br/>

## 🚀 SwanLab: Track, Visualize, and Collaborate on Your Deep Learning Experiments

SwanLab is an open-source, modern tool designed to streamline your deep learning workflows by providing intuitive experiment tracking, visualization, and collaborative features, all in one place. Simplify your machine learning journey by effortlessly integrating with 30+ popular frameworks and running experiments locally or in the cloud. ([Original Repository](https://github.com/SwanHubX/SwanLab))

**Key Features:**

*   **📊 Experiment Tracking and Visualization:** Track metrics, visualize training progress with interactive charts, and gain insights into your model's performance. Supports scalar metrics, images, audio, text, videos, 3D point clouds, and more.

*   **📦 Comprehensive Framework Integrations:** Seamlessly integrate with a wide range of frameworks, including PyTorch, TensorFlow, 🤗 Hugging Face Transformers, PyTorch Lightning, and more.

*   **💻 Hardware Monitoring:** Real-time monitoring of CPU, GPU (Nvidia), NPU (Ascend), and other hardware resources, providing valuable insights into your training environment.

*   **👥 Collaborative Features:** Share experiments, collaborate with your team, and facilitate knowledge sharing to improve team efficiency.

*   **💾 Self-Hosted Option:** Run SwanLab locally or on your own servers for full control and data privacy.

*   **🔌 Plugin Extensibility:** Customize SwanLab with plugins for email notifications, CSV logging, and more.

*   **🆚 Experiment Comparison:** Compare different experiments with interactive tables and charts, and discover new insights.

**Key Benefits:**

*   **Increased Training Efficiency:** Reduce iteration time by gaining quick insights into model performance.
*   **Improved Collaboration:** Enhance teamwork through real-time collaboration.
*   **Better Experiment Management:** Organize and share experimental results across your team.

<br>

## 🌟 Key Updates

Stay up-to-date with the latest features and improvements:

*   **[Recent Updates -  See the latest release notes for feature highlights.](https://github.com/SwanHubX/SwanLab#-%E6%9C%80%E8%BF%91%E6%9B%B4%E6%96%B0)**

<br>

## 🚀 Quickstart

### 1. Installation

```bash
pip install swanlab
```

### 2. Login

1.  [Sign up for a free account](https://swanlab.cn)
2.  Go to your [API Key](https://swanlab.cn/settings) and copy your API Key
3.  Run in your terminal:

```bash
swanlab login
```

Follow the prompts, and paste your API Key.

### 3. Integrate with your code

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

Self-host the Community Edition to manage experiments on your own machine and view results.

*   **[Learn how to use Docker to deploy a self-hosted version here.](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)**

<br>

## 🔥 Examples & Tutorials

Explore practical tutorials and examples to get started:

*   [MNIST Hand-written Digits](https://docs.swanlab.cn/examples/mnist.html)
*   [FashionMNIST Clothes Classification](https://docs.swanlab.cn/examples/fashionmnist.html)
*   [CIFAR-10 Image Classification](https://docs.swanlab.cn/examples/cifar10.html)

**[Find More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)**

<br>

## 🤝 Community

*   **[GitHub Issues](https://github.com/SwanHubX/SwanLab/issues):** For any errors or questions.
*   **[WeChat Community](https://docs.swanlab.cn/guide_cloud/community/online-support.html):** Connect with other SwanLab users.
*   **[Documentation](https://docs.swanlab.cn/):** Comprehensive documentation.

<br>