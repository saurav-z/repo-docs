<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

<br/>

## SwanLab: Open-Source Deep Learning Experiment Tracking & Visualization 🚀

SwanLab is a modern, open-source tool designed to revolutionize deep learning experiment tracking and visualization, providing cloud and offline support with integrations for 30+ major frameworks, making it easy to integrate into your existing projects. [Check out the original repo](https://github.com/SwanHubX/SwanLab).

**Key Features:**

*   **📊 Real-time Tracking & Visualization:** Track and visualize key metrics, hyperparameters, and model outputs in real time.
*   **☁️ Cloud and Offline Support:** Use SwanLab in the cloud (similar to Weights & Biases) or offline for flexibility in any environment.
*   **📦 Comprehensive Integrations:** Seamlessly integrate with over 30 popular frameworks, including PyTorch, TensorFlow, Hugging Face Transformers, and more.
*   **💻 Hardware Monitoring:** Monitor hardware resources (CPU, GPU, memory, disk, network) to optimize training.
*   **📈 Experiment Management:** Efficiently manage experiments with a centralized dashboard for quick overviews and comparisons.
*   **🆚 Experiment Comparison:** Compare results across experiments with tables and charts to identify winning strategies.
*   **👥 Collaboration Features:** Collaborate with your team, share experiments, and provide feedback.
*   **🔌 Plugin Extensibility:** Extend SwanLab's functionality with plugins for custom features and integrations.

**Highlighted Features:**

*   **Enhanced Chart Customization:** Configure chart types, styles, axes, and data sources for optimal visualization.
*   **Comprehensive Data Types:** Supports a wide variety of data types including scalar metrics, images, audio, text, videos, 3D point clouds, and more.
*   **Advanced Table and Text Views:** Utilize interactive tables, markdown rendering, and text navigation for richer insights.
*   **Robust Hardware Monitoring:** Monitor CPU, GPU, memory, and more.

**Recent Updates**

*   **2024.07.17:** Improved line chart configuration, swanlab.Video support, and global chart dashboard enhancements.
*   **2024.07.10:** Enhanced text view with Markdown rendering.
*   **2024.07.06:** Resume functionality, new file recorder, integration with Ray and ROLL frameworks.
*   **2024.06.27:** Chart zoom and smoothing improvements.
*   **2024.06.20:** Integration with Hugging Face Accelerate for distributed training.
*   **2024.06.18:** Integration with AREAL, enhanced curve highlighting.
*   **2024.06.11:** Support for swanlab.echarts.table data type.
*   **2024.06.08:** Local log storage with swanlab sync.
*   **2024.06.01:** Chart drag-and-drop, ECharts custom charts.

<details><summary>Full Changelog</summary>

*   2024.05.25: Support for standard error stream logging, hardware monitoring enhancements.
*   2024.05.14: Experiment Tag support, Log Scale for line charts, grouping drag and drop features.
*   2024.05.09: Line chart creation, data source selection for chart configuration, and GitHub badge generation.
*   2024.04.23: Line chart editing, Kunlunxin XPU hardware detection and monitoring.
*   2024.04.11: Line chart region selection and step range control.
*   2024.04.08: swanlab.Molecule support.
*   2024.04.07: Joint integration with EvalScope.
*   2024.03.30: swanlab.Settings, Cambricon MLU hardware monitoring, and notification plugins.
*   2024.03.21: Hugging Face Transformers integration, Object3D chart, GPU memory and network monitoring.
*   2024.03.12: Private Deployment Release.
*   2024.03.09: Experiment sidebar expansion, Git code button, and MLFlow sync feature.
*   2024.03.06: Joint integration with DiffSynth Studio.
*   2024.03.04: MLFlow Conversion support.
*   2024.03.01: Experiment move function.
*   2024.02.24: Joint integration with EasyR1.
*   2024.02.18: Joint integration with Swift.
*   2024.02.16: Grouping and Grouping creation function.
*   2024.02.09: Joint integration with veRL.
*   2024.02.05: Nested dictionary support and name/notes parameters.
*   2024.01.22: sync_tensorboardX and sync_tensorboard_torch feature.
*   2024.01.17: sync_wandb function.
*   2024.01.11: Performance optimizations in the cloud version.
*   2024.01.01: Line chart smoothing and size adjustment.
*   2023.12.22: Joint integration with LLaMA Factory.
*   2023.12.15: Hardware Monitoring (0.4.0) released.
*   2023.12.06: Integration with LightGBM and XGBoost, increased log line length limit.
*   2023.11.26: Hardware and Cloud provider identification.

</details>

<br>

## Quickstart

### Installation

```bash
pip install swanlab
```

<details><summary>Source Installation</summary>

```bash
git clone https://github.com/SwanHubX/SwanLab.git
pip install -e .
```

</details>

### 2. Login and Get API Key

1.  [Register](https://swanlab.cn) for a free account.
2.  Log in, go to User Settings > [API Key](https://swanlab.cn/settings), and copy your API Key.
3.  Open your terminal and run:

```bash
swanlab login
```

Enter your API Key when prompted.

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

Go to [SwanLab](https://swanlab.cn) to view your experiment.

<br>

## Examples

*   **[ResNet50 Cat/Dog Classification][demo-cats-dogs]**
*   **[Yolov8-COCO128 Object Detection][demo-yolo]**
*   **[Qwen2 Instruction Finetuning][demo-qwen2-sft]**
*   **[LSTM Google Stock Prediction][demo-google-stock]**

[More Examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## Self-Hosting

SwanLab supports self-hosting.

### Docker Deployment

Refer to the documentation: [Docs](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker
```

```bash
./install.sh
```

### Link to Self-Hosted Service

Log in to your self-hosted service:

```bash
swanlab login --host http://localhost:8000
```

<br>

## Integrations

*   [PyTorch](https://docs.swanlab.cn/guide_cloud/integration/integration-pytorch.html)
*   [TensorBoard](https://docs.swanlab.cn/guide_cloud/integration/integration-tensorboard.html)
*   [HuggingFace Transformers](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-transformers.html)

And many more!

## 硬件记录

SwanLab记录AI训练过程中所使用的**硬件信息**和**资源使用情况**，下面是支持情况表格：

| 硬件 | 信息记录 | 资源监控 | 脚本 |
| --- | --- | --- | --- |
| 英伟达GPU | ✅ | ✅ | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| 昇腾NPU | ✅ | ✅ | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| 苹果SOC | ✅ | ✅ | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| 寒武纪MLU | ✅ | ✅ | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| 昆仑芯XPU | ✅ | ✅ | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| 摩尔线程GPU | ✅ | ✅ | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| 沐曦GPU | ✅ | ✅ | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| 海光DCU | ✅ | ✅ | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU     | ✅        | ✅        | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| 内存        | ✅        | ✅        | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| 硬盘        | ✅        | ✅        | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| 网络 | ✅ | ✅ | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

## Plugins

*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [CSV Writer](https://docs.swanlab.cn/plugin/writer-csv.html)

<br>

## Community

*   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues)
*   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html)

<br>

## License

[Apache 2.0 License](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE)

```
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
```

<!-- link -->

[demo-cats-dogs]: https://swanlab.cn/@ZeyiLin/Cats_Dogs_Classification/runs/jzo93k112f15pmx14vtxf/chart
[demo-yolo]: https://swanlab.cn/@ZeyiLin/ultratest/runs/yux7vclmsmmsar9ear7u5/chart
[demo-qwen2-sft]: https://swanlab.cn/@ZeyiLin/Qwen2-fintune/runs/cfg5f8dzkp6vouxzaxlx6/chart
[demo-google-stock]:https://swanlab.cn/@ZeyiLin/Google-Stock-Prediction/charts
[tracking-swanlab-shield-link]:https://swanlab.cn
[tracking-swanlab-shield]: https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg
[visualize-swanlab-shield-link]:https://swanlab.cn
[visualize-swanlab-shield]: https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg