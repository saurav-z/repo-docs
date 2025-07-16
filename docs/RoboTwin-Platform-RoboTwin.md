# RoboTwin: The Premier Bimanual Robotic Manipulation Platform

<p align="center">
    <a href="https://github.com/RoboTwin-Platform/RoboTwin">
        <img src="https://user-images.githubusercontent.com/88101805/463126988-e3ba1575-4411-4a36-ad65-f0b2f49890c3.mp4" alt="RoboTwin Demo" width="60%">
    </a>
</p>

**RoboTwin** is a comprehensive and versatile platform designed for advancing research in bimanual robotic manipulation.  This repository ([Original Repo](https://github.com/RoboTwin-Platform/RoboTwin)) provides a scalable data generator and benchmark with strong domain randomization, making it ideal for developing robust and generalizable robotic manipulation skills.

**Key Features:**

*   **Scalable Data Generation:** Generate diverse and realistic datasets for training and evaluating robotic manipulation algorithms.
*   **Strong Domain Randomization:**  Leverage robust domain randomization techniques to enhance the generalizability of your models.
*   **Bimanual Manipulation Focus:** Specifically designed to address the complexities of dual-arm robotic tasks.
*   **Comprehensive Benchmark:**  Includes a benchmark with a variety of tasks to assess and compare different approaches.
*   **Open-Source and Accessible:**  Freely available under the MIT license for academic and research use.

**Latest Version Information**

*   **RoboTwin 2.0:** A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation ([Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088))

    *   Currently Under Review (2025)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:** ([Technical Report PDF](https://arxiv.org/pdf/2506.23351))

**Important Updates:**

*   **2025/07/09:** Updated endpose control mode.  See the [RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html) for details.
*   **2025/07/08:** Added the [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) branch.
*   **2025/07/02:** Fixed Piper Wrist Bug - see [issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104) - please redownload the embodiment asset.
*   **2025/07/01:** Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21:** Released RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **2025/04/11:** RoboTwin is selected as <i>CVPR Highlight paper</i>!
*   **2025/02/27:** RoboTwin is accepted to <i>CVPR 2025</i> !
*   **2024/09/30:** RoboTwin (Early Version) received <i>the Best Paper Award  at the ECCV Workshop</i>!
*   **2024/09/20:** Officially released RoboTwin.

## Installation

See the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for detailed installation instructions.  The installation process typically takes approximately 20 minutes.

## Tasks Information

Detailed information about the available tasks can be found in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

For in-depth usage guidelines, refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

### Data Collection

While the RoboTwin platform provides over 100,000 pre-collected trajectories in the [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset), we recommend collecting your own data to leverage the platform's high configurability and task diversity.

<img src="./assets/files/domain_randomization.png" alt="Domain Randomization Illustration" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection

Use the following command to run a task and collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Task Configuration

Refer to the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for details on configuring tasks.

## Policy Baselines

### Policies Supported

*   [DP](https://robotwin-platform.github.io/doc/usage/DP.html)
*   [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html)
*   [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html)
*   [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html)
*   [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)
*   [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)

Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, DexVLA, OpenVLA-OFT, SmolVLA, AVR, UniVLA

## Experiment & Leaderboard

> The RoboTwin Platform can be used to explore the following topics:
>
> 1.  Single-task fine-tuning capability
> 2.  Visual robustness
> 3.  Language diversity robustness (language condition)
> 4.  Multi-tasks capability
> 5.  Cross-embodiment performance

Coming soon.

## Citations

If you use this work, please cite the following:

```bibtex
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

```bibtex
@InProceedings{Mu_2025_CVPR,
    author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
    title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27649-27660}
}
```

```bibtex
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

```bibtex
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgements

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos

Code Style: `find . -name "*.py" -exec sh -c 'echo "Processing: {}"; yapf -i --style='"'"'{based_on_style: pep8, column_limit: 120}'"'"' {}' \;`

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is released under the MIT license. See the [LICENSE](./LICENSE) file for more details.