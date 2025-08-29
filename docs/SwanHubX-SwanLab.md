<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br/>

## 🚀 SwanLab: Open-Source Deep Learning Experiment Tracking and Visualization

**SwanLab is an open-source tool designed to streamline your deep learning workflow by providing intuitive experiment tracking, visualization, and collaboration features.** ([Original Repo](https://github.com/SwanHubX/SwanLab))

**Key Features:**

*   📊 **Experiment Tracking & Visualization:** Track metrics, hyperparameters, and visualize training progress with interactive charts.  Supports scalars, images, audio, text, videos, 3D point clouds, and more.
*   💻 **Framework Integrations:** Seamlessly integrates with over 30 popular frameworks including PyTorch, TensorFlow, Hugging Face Transformers, and more.
*   ⚙️ **Hardware Monitoring:** Real-time monitoring of CPU, GPU, memory, disk, and network usage.
*   📦 **Experiment Management:**  A centralized dashboard for managing projects and experiments.
*   🆚 **Comparison:** Compare results and hyperparameters across experiments.
*   🤝 **Collaboration:**  Facilitate team training by sharing experiment links and real-time updates.
*   ☁️ **Cloud & Self-Hosted:** Use SwanLab online or self-host the open-source version for offline access.
*   🔌 **Plugin Extensibility:** Extend functionality with plugins for features like notifications, custom logging, and more.

![swanlab-overview](readme_files/swanlab-overview.png)

**Core Features:**

-   **Comprehensive Visualization:** Display training metrics, hyperparameters, and system metrics in rich, interactive charts.
-   **Framework Compatibility:** Easily integrates with popular deep learning frameworks.
-   **Flexible Deployment:** Utilize cloud-based access or self-host SwanLab.

**Latest Updates:**

*   See the full [更新日志](https://docs.swanlab.cn/zh/guide_cloud/general/changelog.html)

<br/>

## Getting Started

### 1. Installation

```bash
pip install swanlab
```

### 2. Login and Get API Key
1.  [Register](https://swanlab.cn)
2.  Copy your API key: User Settings -> [API Key](https://swanlab.cn/settings)
3.  Run `swanlab login` in your terminal and enter your API key.

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

Access your first SwanLab experiment on the [SwanLab](https://swanlab.cn) platform.

<br/>

## 💻 Self-Hosting

Self-hosting community version for offline use.

![swanlab-docker](readme_files/swanlab-docker.png)

### 1. Deploy with Docker

See: [Doc](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

China fast install:

```bash
./install.sh
```

From DockerHub:

```bash
./install-dockerhub.sh
```

### 2. Direct your experiments to your service

Login to the self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

<br/>

## 🔥 Real-World Examples

*   [Demo Projects & Tutorials](https://docs.swanlab.cn/zh/examples/mnist.html)
*   [Awesome Tutorials](https://github.com/SwanHubX/SwanLab#-%E5%AE%9E%E6%88%98%E6%A1%88%E4%BE%8B)

<br/>

## 🎮 Hardware Monitoring

SwanLab tracks **hardware usage** during training.

| Hardware     | Information Tracking | Resource Monitoring | Script                                                                     |
| ------------ | ------------------- | ------------------- | -------------------------------------------------------------------------- |
| Nvidia GPU   | ✅                   | ✅                   | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py)      |
| Ascend NPU   | ✅                   | ✅                   | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py)      |
| Apple SOC    | ✅                   | ✅                   | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py)      |
| Cambricon MLU | ✅                   | ✅                   | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py)  |
| Kunlunxin XPU| ✅                   | ✅                   | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| MooreThreads GPU| ✅                  | ✅                   | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| Metax GPU  | ✅                   | ✅                   | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU  | ✅                   | ✅                   | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU          | ✅                   | ✅                   | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py)             |
| Memory       | ✅                   | ✅                   | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py)        |
| Disk         | ✅                   | ✅                   | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py)            |
| Network      | ✅                   | ✅                   | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py)        |

<br/>

## 🚗 Framework Integrations

Integrate your favorite frameworks with SwanLab!

*   [List of Supported Frameworks](https://github.com/SwanHubX/SwanLab#-%E6%A1%86%E6%9E%B6%E9%9B%86%E6%88%90)
*   [Framework Integration Guides](https://docs.swanlab.cn/guide_cloud/integration/)

<br/>

## 🔌 Plugins & APIs

Enhance your experiment management with plugins.

*   [Create Custom Plugins](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Plugin List](https://github.com/SwanHubX/SwanLab#-%E6%8F%92%E4%BB%B6%E4%B8%8Eapi)
*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br/>

## 🆚 Comparison

SwanLab provides a collaborative and flexible platform for experiment tracking, visualization, and collaboration.

*   **[Tensorboard](https://github.com/tensorflow/tensorboard):** SwanLab extends Tensorboard functionality, offering cloud-based experiment synchronization, collaboration, and advanced data management.
*   **[Weights & Biases](https://wandb.ai/site):** SwanLab is a free, open-source alternative to Weights & Biases, offering self-hosting and customizable features.

<br/>

## 👥 Community

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues)
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html)
*   [Email Support](zeyi.lin@swanhub.co)

<br/>

## 📃 License

This project is licensed under the [Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE).