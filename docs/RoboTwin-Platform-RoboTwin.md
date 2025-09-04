# RoboTwin: The Ultimate Bimanual Robotic Manipulation Platform

RoboTwin is a comprehensive platform designed to advance bimanual robotic manipulation, offering a scalable data generator, benchmark, and strong domain randomization capabilities. **Explore RoboTwin on GitHub: [https://github.com/RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)**.

## Key Features

*   **Scalable Data Generation:** Generate large-scale, diverse datasets for training and evaluating robotic manipulation algorithms.
*   **Robust Domain Randomization:** Employ advanced domain randomization techniques to enhance the generalizability of your models.
*   **Comprehensive Benchmark:** Evaluate and compare your models against state-of-the-art baselines on a variety of challenging tasks.
*   **Bimanual Robotic Manipulation:** Focus specifically on the complexities and benefits of dual-arm robotic systems.
*   **Pre-collected Dataset:** Access a large, pre-collected dataset of over 100,000 trajectories to get started quickly.
*   **Modular Design:** Easily integrate with existing robotics frameworks and algorithms.
*   **Active Community:** Join a growing community of researchers and developers.

## What's New in RoboTwin 2.0

RoboTwin 2.0 offers significant improvements, including:

*   **Enhanced Domain Randomization:** Improved realism and robustness in simulated environments.
*   **Extended Task Suite:** Expanded set of manipulation tasks to challenge your algorithms.
*   **Leaderboard:** Track and compare performance on the RoboTwin leaderboard: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **Updated Documentation:** Easier to understand documentation
*   **Technical Report:** "Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop" [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)

## Versions & Resources

*   **Latest Version:** RoboTwin 2.0
    *   [Webpage](https://robotwin-platform.github.io/)
    *   [Document](https://robotwin-platform.github.io/doc/)
    *   [Paper (Under Review)](https://arxiv.org/abs/2506.18088)
    *   [Community](https://robotwin-platform.github.io/doc/community/index.html)
    *   [Leaderboard](https://robotwin-platform.github.io/leaderboard)

*   **Earlier Versions:** Access previous versions and related papers for in-depth understanding. See the original README for these links.

## Installation

Get started with RoboTwin:

1.  Refer to the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for detailed installation instructions.
2.  Installation takes approximately 20 minutes.

## Tasks & Usage

### Task Information

Explore various bimanual manipulation tasks.  See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

### Data Collection

*   Collect data by executing tasks using a variety of setups.
*   Data collection can be done using the following command:
    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

### Configuration

*   Customize tasks by modifying configuration files.
*   Refer to the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for details.

## Policy Baselines

RoboTwin supports various policy baselines: [DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html), [OpenVLA-oft](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html), [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html), [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html)

Deploy your own policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

RoboTwin is designed for exploring various research topics:

*   Single-task fine-tuning.
*   Visual robustness.
*   Language diversity robustness (language condition).
*   Multi-task capabilities.
*   Cross-embodiment performance.

Access the full leaderboard and settings: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Dataset

Utilize the large-scale pre-collected dataset available at: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

Please cite the following if you use RoboTwin in your research:

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

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos.

## Contact

Contact [Tianxing Chen](https://tianxingchen.github.io) for questions or suggestions.

## License

Released under the MIT License.  See [LICENSE](./LICENSE) for details.