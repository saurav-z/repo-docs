# RoboTwin: A Benchmark and Scalable Data Generator for Bimanual Robotic Manipulation

RoboTwin provides a comprehensive platform for research and development in bimanual robotic manipulation, offering a scalable data generator, benchmark, and strong domain randomization capabilities.  [Explore the RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Scalable Data Generation:** Generate diverse and realistic training data for bimanual robotic tasks.
*   **Strong Domain Randomization:**  Improve robustness and generalizability through extensive domain randomization techniques.
*   **Bimanual Manipulation Tasks:** Supports a wide range of challenging bimanual manipulation tasks.
*   **Pre-collected Large-scale Dataset:** Includes a large, pre-collected dataset for faster experimentation.
*   **Policy Baselines:**  Offers implementations of various policy baselines for benchmarking, including DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA.
*   **Leaderboard:** Track and compare performance on various tasks using the RoboTwin leaderboard.
*   **Comprehensive Documentation:**  Detailed documentation to guide users through installation, usage, and task implementation.
*   **Active Community:**  Engage with a community of researchers and developers.

## What's New:

*   **RoboTwin 2.0 is Released!** Includes significant updates to the platform and tasks.  [Webpage](https://robotwin-platform.github.io/), [Document](https://robotwin-platform.github.io/doc), [Paper](https://arxiv.org/abs/2506.18088), [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **CVPR 2025 Highlight Paper:** RoboTwin has been selected as a highlight paper at CVPR 2025!
*   **CVPR 2025 Challenge:** RoboTwin Dual-Arm Collaboration Challenge @ CVPR '25 MEIS Workshop is now available.  [Technical Report](https://arxiv.org/abs/2506.23351)
*   **ECCV 2024 Best Paper Award:** RoboTwin (Early Version) received the Best Paper Award at the ECCV Workshop.

## Quick Start

### Installation:
See [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for detailed installation instructions.

### Tasks Information
See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

### Usage:

1.  **Modify Task Config:**  Refer to [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

2.  **Data Collection**:
    Collect data by running the provided scripts.
    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

## Experiment & Leaderboard

> We recommend that the RoboTwin Platform can be used to explore the following topics: 
> 1. single - task fine - tuning capability
> 2. visual robustness
> 3. language diversity robustness (language condition)
> 4. multi-tasks capability
> 5. cross-embodiment performance

The full leaderboard and setting can be found in: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Dataset

Access the pre-collected large-scale dataset at [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```
```
@InProceedings{Mu_2025_CVPR,
    author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
    title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27649-27660}
}
```

```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgements

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos.

## License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

---

**Note:**  This improved README is optimized for search engines, providing clear headings, bullet points, and keywords related to bimanual robotic manipulation and the RoboTwin platform.  It also includes important links and citations.