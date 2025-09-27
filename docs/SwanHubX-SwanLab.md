<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="readme_files/swanlab-logo-type2-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="readme_files/swanlab-logo-type2-light.svg">
  <img alt="SwanLab" src="readme_files/swanlab-logo-type2-light.svg" width="300" height="130">
</picture>

</div>

## 🚀 Supercharge Your Deep Learning Experiments with SwanLab

SwanLab is an open-source, user-friendly tool for tracking, visualizing, and collaborating on your deep learning experiments. Easily integrate with your code and unlock powerful insights! Visit the [original repo](https://github.com/SwanHubX/SwanLab) to get started.

**Key Features:**

*   📊 **Intuitive Visualization:** Visualize your training progress with interactive charts and graphs.
*   📝 **Automated Logging:** Automatically logs metrics, hyperparameters, and system information.
*   💻 **Hardware Monitoring:** Track CPU, GPU, memory, and disk usage in real-time.
*   🤝 **Collaborative Experimentation:** Share experiments, compare results, and collaborate with your team.
*   ☁️ **Cloud and Self-Hosted Options:** Use SwanLab online or deploy it on your own infrastructure.
*   🔌 **Extensive Framework Integrations:** Supports 30+ popular frameworks including PyTorch, Transformers, and more.
*   ✅ **Comprehensive Experiment Management:** Organize and analyze your experiments with ease.

<br>

## Main Features

*   **Experiment Tracking & Visualization:**
    *   Real-time tracking of metrics, hyperparameters, and system resources.
    *   Interactive charts for visualizing training curves, loss functions, and other key performance indicators.
    *   Support for various data types: scalar metrics, images, audio, text, videos, 3D point clouds, and custom ECharts.
*   **Deep Integration:**
    *   Seamless integration with popular deep learning frameworks like PyTorch, TensorFlow, and more.
    *   Automatic logging of hyperparameters, model configurations, and system information.
*   **Flexible Deployment:**
    *   Cloud-based platform for easy access and collaboration.
    *   Self-hosted option for on-premise deployment and data privacy.
*   **Collaboration and Sharing:**
    *   Share your experiments with colleagues and collaborators.
    *   Compare different experiments and analyze results side-by-side.
    *   Multi-user collaboration with organization and project management.
*   **Hardware Monitoring:**
    *   Real-time monitoring of CPU, GPU, memory, and disk usage.
    *   Comprehensive hardware information for debugging and performance analysis.
*   **Other features:**
    *   Support for resume training.
    *   Comprehensive experiment management with a dedicated dashboard.
    *   Sharing of results via persistent URLs.
    *   Plugin support for extending functionality with features like Slack and Discord notifications, and CSV logging.

<br>

## What's New

### Recent Updates

*   **2025.09.22:** 📊 New UI launch; Table view supports global sorting and filtering; Data-level unification of table view and chart view;
*   **2025.09.12:** 🔢 Support for creating **scalar charts**, flexibly displaying statistical values of experimental metrics; Major upgrade of the organization management page, providing more powerful permission control and project management capabilities;
*   **2025.08.19:** 🤔 More powerful chart rendering performance and low-invasive loading animation, allowing researchers to focus more on experiment analysis itself; Integrated the excellent [MLX-LM](https://github.com/ml-explore/mlx-lm), [SpecForge](https://github.com/sgl-project/SpecForge) frameworks, providing more training experiences for more scenarios;
*   **2025.08.06:** 👥 **Training Light Collaboration** is online, supporting invitation of project collaborators, sharing project links and QR codes; Workspace supports list view, supports displaying project Tags;
*   **2025.07.29:** 🚀 **Experiment filtering and sorting** supported in the sidebar; 📊 Table view goes online with **column control panel**, enabling convenient hiding and display of columns; 🔐 **Multi-API Key** management goes online to make your data more secure; swanlab sync improves compatibility with log file integrity, adapting to scenarios such as training crashes; New charts - PR curve, ROC curve, confusion matrix are online, [documentation](https://docs.swanlab.cn/api/py-pr_curve.html);

<details><summary>View full changelog</summary>

*   **2025.07.17:** 📊 More powerful **line chart configuration**, supporting flexible configuration of line type, color, thickness, grid, and legend position, etc.; 📹 Supports **swanlab.Video** data type, enabling recording and visualization of GIF format files; Global chart dashboard supports configuring Y-axis and the maximum number of experiments displayed;
*   **2025.07.10:** 📚 More powerful **text view**, supporting Markdown rendering and arrow key switching, can be created by `swanlab.echarts.table` and `swanlab.Text`, [Demo](https://swanlab.cn/@ZeyiLin/ms-swift-rlhf/runs/d661ty9mslogsgk41fp0p/chart)
*   **2025.07.06:** 🚄 Supports **resume breakpoint training**; New plugin **file recorder**; Integrated [ray](https://github.com/ray-project/ray) framework, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-ray.html); Integrated [ROLL](https://github.com/volcengine/ROLL) framework, thanks [@PanAndy](https://github.com/PanAndy), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-roll.html)
*   **2025.06.27:** 📊 Supports **local magnification of small line charts**; Supports configuration of **single line chart smoothing**; Greatly improved the interaction effect after image charts are enlarged;
*   **2025.06.20:** 🤗 Integrated [accelerate](https://github.com/huggingface/accelerate) framework, [PR](https://github.com/huggingface/accelerate/pull/3605), [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-accelerate.html), enhancing the experimental recording experience in distributed training;
*   **2025.06.18:** 🐜 Integrated [AREAL](https://github.com/inclusionAI/AReaL) framework, thanks [@xichengpro](https://github.com/xichengpro), [PR](https://github.com/inclusionAI/AReaL/pull/98), [documentation](https://inclusionai.github.io/AReaL/tutorial/quickstart.html#monitoring-the-training-process); 🖱 Supports highlighting the corresponding curve when hovering the experiment in the sidebar; Supports cross-group comparison of line charts; Supports setting experiment name cropping rules;
*   **2025.06.11:** 📊 Supports **swanlab.echarts.table** data type, supports pure text chart display; Supports **stretching interaction** for grouping to increase the number of charts displayed simultaneously; Table view adds **metric maximum/minimum** options;
*   **2025.06.08:** ♻️ Supports storing complete experimental log files locally, uploading local log files to cloud/privately deployed ends through **swanlab sync**; Hardware monitoring supports **Hygon DCU**;
*   **2025.06.01:** 🏸 Supports **free drag and drop of charts**; Supports **ECharts custom charts**, adding more than 20 chart types including bar charts, pie charts, and histograms; Hardware monitoring supports **MetaX GPU**; Integrated **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** framework;
*   **2025.05.25:** The logs support recording **standard error streams**, and the printing information of frameworks like PyTorch Lightning can be better recorded; Hardware monitoring supports **Moore Threads**; Added a security protection function for recording running commands, and the API Key will be automatically hidden;
*   **2025.05.14:** Supports **Experiment Tag**; Supports line chart **Log Scale**; Supports **Group Dragging**; Greatly optimizes the experience of uploading a large number of metrics; Added `swanlab.OpenApi` open interface;
*   **2025.05.09:** Supports **line chart creation**; Configuration chart function adds **data source selection** function, supporting different metrics displayed on a single chart; Supports generating **training project GitHub badge**;
*   **2025.04.23:** Supports line chart **editing**, supports freely configuring the X and Y axis data ranges and title styles of the chart; Chart search supports **regular expressions**; Supports hardware detection and monitoring of **Kunlunxin XPU**;
*   **2025.04.11:** Supports line chart **local area selection**; Supports global selection of the step range of the dashboard line chart; Supports one-click hiding of all charts;
*   **2025.04.08:** Supports **swanlab.Molecule** data type, supporting the recording and visualization of biochemical molecular data; Supports saving the sorting, filtering, and column order change status in the table view;
*   **2025.04.07:** We have completed the joint integration with [EvalScope](https://github.com/ModelScope/EvalScope), and now you can use SwanLab in EvalScope to **evaluate the performance of large models**;
*   **2025.03.30:** Supports the **swanlab.Settings** method, supporting more refined experimental behavior control; Supports **Cambricon MLU** hardware monitoring; Supports [Slack notifications](https://docs.swanlab.cn/plugin/notification-slack.html), [Discord notifications](https://docs.swanlab.cn/plugin/notification-discord.html);
*   **2025.03.21:** 🎉🤗HuggingFace Transformers has officially integrated SwanLab (>=4.50.0 version), [#36433](https://github.com/huggingface/transformers/pull/36433); New **Object3D chart**, supporting recording and visualization of three-dimensional point clouds, [documentation](https://docs.swanlab.cn/api/py-object3d.html); Hardware monitoring supports the recording of GPU video memory (MB), disk utilization, and network up and down;
*   **2025.03.12:** 🎉🎉SwanLab **private deployment version** is now released!! [🔗Deployment Documentation](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html); SwanLab now supports plugin extensions, such as [Email notifications](https://docs.swanlab.cn/plugin/notification-email.html), [Feishu notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   **2025.03.09:** Supports **expanding the experimental sidebar**; Added the Git code button; Added the **sync_mlflow** function, supporting synchronizing experimental tracking with the mlflow framework;
*   **2025.03.06:** We have completed joint integration with [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio), and now you can use SwanLab in DiffSynth Studio to **track and visualize Diffusion model text-to-image/video experiments**, [Usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-diffsynth-studio.html);
*   **2025.03.04:** Added the **MLFlow conversion** function, supporting converting MLFlow experiments to SwanLab experiments, [Usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-mlflow.html);
*   **2025.03.01:** Added the **Move Experiment** function, now you can move experiments to different projects in different organizations;
*   **2025.02.24:** We have completed joint integration with [EasyR1](https://github.com/hiyouga/EasyR1), and now you can use SwanLab in EasyR1 to **track and visualize multi-modal large model reinforcement learning experiments**, [Usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-easyr1.html)
*   **2025.02.18:** We have completed joint integration with [Swift](https://github.com/modelscope/ms-swift), and now you can use SwanLab in the CLI/WebUI of Swift to **track and visualize large model fine-tuning experiments**, [Usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-swift.html).
*   **2025.02.16:** Added the **Move Group, Create Group** function for charts.
*   **2025.02.09:** We have completed joint integration with [veRL](https://github.com/volcengine/verl), and now you can use SwanLab in veRL to **track and visualize large model reinforcement learning experiments**, [Usage guide](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html).
*   **2025.02.05:** `swanlab.log` supports nested dictionaries [#812](https://github.com/SwanHubX/SwanLab/pull/812), adapting to Jax framework features; supports `name` and `notes` parameters;
*   **2025.01.22:** Added the `sync_tensorboardX` and `sync_tensorboard_torch` functions, supporting synchronization of experimental tracking with these two TensorBoard frameworks;
*   **2025.01.17:** Added the `sync_wandb` function, [documentation](https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html), supports synchronization of experiment tracking with Weights & Biases; greatly improved log rendering performance
*   **2025.01.11:** The cloud version greatly optimizes the performance of the project table and supports drag and drop, sorting, and filtering interactions
*   **2025.01.01:** Added line chart **persistent smoothing**, line chart drag and drop to change size, and optimized chart browsing experience
*   **2024.12.22:** We have completed joint integration with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), and now you can use SwanLab in LLaMA Factory to **track and visualize large model fine-tuning experiments**, [Usage guide](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#use-swanlab-logger).
*   **2024.12.15:** **Hardware Monitoring (0.4.0)** function launched, supporting system-level information recording and monitoring for CPU, NPU (Ascend), and GPU (Nvidia).
*   **2024.12.06:** Added integration for [LightGBM](https://docs.swanlab.cn/guide_cloud/integration/integration-lightgbm.html) and [XGBoost](https://docs.swanlab.cn/guide_cloud/integration/integration-xgboost.html); Increased the limit on the length of a single log line.
*   **2024.11.26:** Environment tab - hardware section supports identifying **Huawei Ascend NPU** and **Kunpeng CPU**; The cloud vendor section supports identifying QingCloud **Jishi Zhisuan**.
</details>

<br>

## Quick Start

1.  **Installation**

    ```bash
    pip install swanlab
    ```

    <details><summary>Source Installation</summary>

    ```bash
    git clone https://github.com/SwanHubX/SwanLab.git
    pip install -e .
    ```

    </details>

2.  **Login**
    *   Register a free account: [SwanLab](https://swanlab.cn)
    *   Log in and copy the API Key from your user settings [API Key](https://swanlab.cn/settings)
    *   Run the following command in your terminal:

        ```bash
        swanlab login
        ```

        Then, enter your API Key.

3.  **Integrate SwanLab with your code:**

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

    Congratulations! You can now view your experiment on [SwanLab](https://swanlab.cn).

<br>

## Self-Hosting

SwanLab supports self-hosting for offline use.

### 1. Deploying with Docker

See instructions [here](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html).

```bash
git clone https://github.com/SwanHubX/self-hosted.git
cd self-hosted/docker

# For China users
./install.sh

# From DockerHub
./install-dockerhub.sh
```

### 2.  Experiment Setup

Log in to the self-hosted service

```bash
swanlab login --host http://localhost:8000
```

After logging in, your experiments will be recorded on the self-hosted service.

<br>

##  Example Cases

*   **[ResNet50 猫狗分类][demo-cats-dogs]**: Track image classification on a cats and dogs dataset.
*   **[Yolov8-COCO128 目标检测][demo-yolo]**: Object detection using Yolov8.
*   **[Qwen2 指令微调][demo-qwen2-sft]**: Instruction fine-tuning of Qwen2 language models.
*   **[LSTM Google 股票预测][demo-google-stock]**: Predict Google stock prices with an LSTM model.

[More examples](https://docs.swanlab.cn/zh/examples/mnist.html)

<br>

## Hardware Monitoring

SwanLab records the hardware information and resource usage during the training process:

| Hardware    | Information Recording | Resource Monitoring | Script                                                   |
| :---------- | :------------------- | :------------------- | :------------------------------------------------------- |
| NVIDIA GPU  | ✅                   | ✅                   | [nvidia.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/nvidia.py) |
| Ascend NPU  | ✅                   | ✅                   | [ascend.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/npu/ascend.py) |
| Apple SOC   | ✅                   | ✅                   | [apple.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/soc/apple.py) |
| Cambricon MLU| ✅                   | ✅                   | [cambricon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/mlu/cambricon.py) |
| Kunlunxin XPU| ✅                   | ✅                   | [kunlunxin.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/xpu/kunlunxin.py) |
| Moore Threads GPU| ✅                   | ✅                   | [moorethreads.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/moorethreads.py) |
| MetaX GPU| ✅                   | ✅                   | [metax.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/gpu/metax.py) |
| Hygon DCU   | ✅                   | ✅                   | [hygon.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/dcu/hygon.py) |
| CPU         | ✅                   | ✅                   | [cpu.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/cpu.py) |
| Memory      | ✅                   | ✅                   | [memory.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/memory.py) |
| Disk        | ✅                   | ✅                   | [disk.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/disk.py) |
| Network    | ✅                   | ✅                   | [network.py](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/run/metadata/hardware/network.py) |

<br>

## Framework Integrations

Integrate SwanLab with your favorite frameworks!

*   **Basic Frameworks:** PyTorch, MindSpore, Keras.
*   **Specialized/Fine-tuning Frameworks:** PyTorch Lightning, HuggingFace Transformers, LLaMA Factory, Modelscope Swift, DiffSynth Studio, Sentence Transformers, PaddleNLP, OpenMind, Torchtune, XTuner, MMEngine, FastAI, LightGBM, XGBoost, MLX-LM.
*   **Evaluation Frameworks:** EvalScope.
*   **Computer Vision Frameworks:** Ultralytics, MMDetection, MMSegmentation, PaddleDetection, PaddleYOLO.
*   **Reinforcement Learning Frameworks:** Stable Baseline3, veRL, HuggingFace trl, EasyR1, AReaL, ROLL.
*   **Other Frameworks:** Tensorboard, Weights&Biases, MLFlow, HuggingFace Accelerate, Ray, Unsloth, Hydra, Omegaconf, OpenAI, ZhipuAI.

<br>

## Plugins & API

Expand SwanLab's functionality with plugins!

*   [Customize Your Plugin](https://docs.swanlab.cn/plugin/custom-plugin.html)
*   [Email Notifications](https://docs.swanlab.cn/plugin/notification-email.html)
*   [Feishu Notifications](https://docs.swanlab.cn/plugin/notification-lark.html)
*   [DingTalk Notifications](https://docs.swanlab.cn/plugin/notification-dingtalk.html)
*   [Enterprise WeChat Notifications](https://docs.swanlab.cn/plugin/notification-wxwork.html)
*   [Discord Notifications](https://docs.swanlab.cn/plugin/notification-discord.html)
*   [Slack Notifications](https://docs.swanlab.cn/plugin/notification-slack.html)
*   [CSV Logger](https://docs.swanlab.cn/plugin/writer-csv.html)
*   [File Logger](https://docs.swanlab.cn/plugin/writer-filelogdir.html)

Open API:

*   [OpenAPI](https://docs.swanlab.cn/api/py-openapi.html)

<br>

## Comparison: Tensorboard vs. Weights & Biases vs. SwanLab

*   **Tensorboard vs SwanLab:** SwanLab offers online collaboration, persistent dashboards, and superior table functionality for experiment management.
*   **Weights & Biases vs. SwanLab:** SwanLab provides open-source, free, and self-hosted options, giving you more control over your data and experimentation.

<br>

## Community

*   **Community & Support:**
    *   [GitHub Issues](https://github.com/SwanHubX/SwanLab/issues): For any errors and issues.
    *   [Email Support](zeyi.lin@swanhub.co): For any feedback.
    *   [WeChat Group](https://docs.swanlab.cn/guide_cloud/community/online-support.html): Join us to discuss SwanLab usage.

*   **README Badges:**  Add SwanLab badges to your project!

    ```
    [![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge2.svg)](your experiment url)
    [![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](your experiment url)
    ```

    More design assets: [assets](https://github.com/SwanHubX/assets)

*   **Citing SwanLab:**

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

*   **Contributing:**  Read the [Contribution Guide](CONTRIBUTING.md).

<br>

**Contributors**

<a href="https://github.com/swanhubx/swanlab/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=swanhubx/swanlab" />
</a>

<br>

<img src="./readme_files/swanlab-and-user.png" width="50%" />

<br>

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