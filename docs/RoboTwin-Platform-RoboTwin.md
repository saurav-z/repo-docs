# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

**RoboTwin** is a cutting-edge platform for bimanual robotic manipulation research, offering a scalable data generator, robust domain randomization, and a comprehensive benchmark to push the boundaries of dual-arm robotics. Explore the RoboTwin platform on [GitHub](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Advanced Bimanual Tasks:** RoboTwin provides a diverse set of challenging manipulation tasks designed to test the capabilities of dual-arm robotic systems.
*   **Scalable Data Generation:**  Generate massive datasets with ease to train and evaluate your robotic manipulation algorithms.
*   **Strong Domain Randomization:** Utilize robust domain randomization techniques to improve the generalization ability of your models across different environments and scenarios.
*   **Comprehensive Benchmark:** Evaluate your algorithms using the RoboTwin benchmark, which includes a leaderboard and various evaluation metrics.
*   **Multiple Versions & Leaderboards:**  Explore and benchmark with different RoboTwin versions and official leaderboards.

## Latest Updates:

*   **RoboTwin 2.0 (Latest Version):**  Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation.

    *   [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:** Technical Report available.
    *   [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)

## Why RoboTwin?

RoboTwin empowers researchers and developers to advance the state-of-the-art in bimanual robotic manipulation by providing a realistic, scalable, and challenging environment.  It's an excellent resource for:

*   Researchers exploring new algorithms and techniques for dual-arm robotics.
*   Developers seeking to benchmark their systems and improve their performance.
*   Anyone interested in pushing the boundaries of robotic manipulation.

## Getting Started:

### Installation:

See the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for detailed installation instructions. Installation takes approximately 20 minutes.

### Tasks:

Explore a diverse set of tasks in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

### Usage:

*   **Document:** [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html)
*   **Data Collection:**  Collect your own data for tasks and embodiments:

    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```
*   **Modify Task Config:**  Refer to [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines:

RoboTwin supports a variety of policy baselines.  See the usage documentation for details.
*   **Policies Supported:** DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, LLaVA-VLA, and more.
*   **Deploy Your Policy:** [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

Explore and contribute to the RoboTwin Leaderboard:  [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).
  > We recommend that the RoboTwin Platform can be used to explore the following topics: 
  > 1. single - task fine - tuning capability
  > 2. visual robustness
  > 3. language diversity robustness (language condition)
  > 4. multi-tasks capability
  > 5. cross-embodiment performance
## Pre-collected Large-scale Dataset:

Access a pre-collected dataset on Hugging Face: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations:

If you find RoboTwin useful, please cite the following papers:

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

## Acknowledgements:

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos

## Contact:

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License:

This project is released under the MIT license. See the [LICENSE](./LICENSE) file for details.