# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

**Explore cutting-edge research and advancements in dual-arm robotic manipulation with RoboTwin, a comprehensive platform for scalable data generation, robust benchmarking, and collaborative robotics challenges.  [Visit the original repository](https://github.com/RoboTwin-Platform/RoboTwin) to get started!**

## Key Features:

*   **Scalable Data Generation:** Create diverse datasets for training and evaluating robotic manipulation skills.
*   **Strong Domain Randomization:**  Enhance the robustness of your models with advanced domain randomization techniques.
*   **Robust Benchmarking:**  Test and compare your algorithms using a standardized benchmark environment.
*   **Dual-Arm Collaboration Challenges:** Participate in and develop solutions for challenging bimanual manipulation tasks.
*   **Pre-collected Large-scale Dataset:** Utilize the RoboTwin 2.0 Dataset for efficient data-driven research [Hugging Face Dataset Link](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
*   **Policy Baselines:** Explore and deploy a variety of policy baselines:  DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, LLaVA-VLA.

##  What's New in RoboTwin 2.0?

RoboTwin 2.0 introduces significant upgrades and features, building upon the successes of previous versions.

*   **Latest Version: RoboTwin 2.0** (Under Review 2025) [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)

*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop**
    *   Official Technical Report: [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)

*   **RoboTwin 1.0** (Accepted to CVPR 2025 - Highlight)
    *   [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)

*   **RoboTwin Early Version** (Accepted to ECCV Workshop 2024 - Best Paper Award)
    *   [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Updates

*   **2025/08/06**: RoboTwin 2.0 Leaderboard released: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **2025/07/23**: RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/19**: Fixed DP3 evaluation code error.
*   **2025/07/09**: Updated endpose control mode. See [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)] for details.
*   **2025/07/08**: Released [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch.
*   **2025/07/02**: Fixed Piper Wrist Bug.  Please redownload the embodiment asset.
*   **2025/07/01**: Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21**: Released RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **2025/04/11**: RoboTwin selected as a *CVPR Highlight paper*!
*   **2025/02/27**: RoboTwin accepted to *CVPR 2025*!
*   **2024/09/30**: RoboTwin (Early Version) received the Best Paper Award at the ECCV Workshop!
*   **2024/09/20**: Officially released RoboTwin.

## Installation

Get started by following the installation instructions: [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation takes about 20 minutes.

## Tasks Information

Explore the available tasks and their details: [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html)

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Document
Refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for comprehensive details.

### Data Collection

RoboTwin encourages users to perform their own data collection for maximum flexibility and diversity.

<img src="./assets/files/domain_randomization.png" alt="Domain Randomization" style="display: block; margin: auto; width: 100%;">

### Task Running and Data Collection

Run the following command to collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Modify Task Config

Customize your tasks using the configurations documented here: [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html)

## Experiment & Leaderboard

We encourage exploration of the following areas with the RoboTwin platform:

*   Single-task fine-tuning capabilities
*   Visual robustness
*   Language diversity robustness (language condition)
*   Multi-task capabilities
*   Cross-embodiment performance

Find the full leaderboard and settings: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard)

## Citations

If you find RoboTwin valuable for your research, please cite our work:

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

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos

## Contact

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.