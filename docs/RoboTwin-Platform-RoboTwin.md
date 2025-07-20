# RoboTwin: A Scalable Benchmark for Bimanual Robotic Manipulation

<p align="center">
  <a href="https://github.com/RoboTwin-Platform/RoboTwin">
    <img src="./assets/files/50_tasks.gif" width="50%">
  </a>
</p>

**RoboTwin is a comprehensive platform designed to advance the field of bimanual robotic manipulation, providing a scalable data generator and benchmark with strong domain randomization.** Explore cutting-edge research and contribute to the future of robotics! ([Original Repository](https://github.com/RoboTwin-Platform/RoboTwin))

## Key Features:

*   **Scalable Data Generation:** Generate diverse and realistic datasets for training and evaluating robotic manipulation algorithms.
*   **Strong Domain Randomization:** Robust domain randomization techniques to enhance the generalizability of learned policies.
*   **Bimanual Robotic Manipulation:** Focus on complex tasks involving two-arm coordination.
*   **Pre-collected Trajectories:** Access over 100,000 pre-collected trajectories via the [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
*   **Multiple Versions:** Explore different stages of development with [RoboTwin 2.0](https://robotwin-platform.github.io/), [RoboTwin 1.0](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0) and Early Versions.
*   **Active Development:** The platform is continuously updated with new features, tasks, and baselines.

## What's New

### RoboTwin 2.0
*   **[RoboTwin 2.0](https://robotwin-platform.github.io/) Release!**
    *   [Webpage](https://robotwin-platform.github.io/)
    *   [Document](https://robotwin-platform.github.io/doc)
    *   [Paper](https://arxiv.org/abs/2506.18088)
*   **Challenge-Cup-2025 Branch**  (第十九届挑战杯分支)
*   **RoboTwin Dual-Arm Collaboration Challenge Technical Report** at CVPR 2025 MEIS Workshop:
    *   [Technical Report](https://arxiv.org/pdf/2506.23351)

### Previous Versions
*   **RoboTwin 1.0:** Accepted to <i style="color: red; display: inline;"><b>CVPR 2025 (Highlight)</b></i>
    *   [PDF](https://arxiv.org/pdf/2504.13059)
    *   [arXiv](https://arxiv.org/abs/2504.13059)
*   **RoboTwin Early Version:** Received <i style="color: red; display: inline;"><b>Best Paper Award at ECCV Workshop 2024</b></i>
    *   [PDF](https://arxiv.org/pdf/2409.02920)
    *   [arXiv](https://arxiv.org/abs/2409.02920)

## Installation

Follow the instructions in the RoboTwin 2.0 Document for installation.
[Installation Instructions](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). (Takes about 20 minutes).

## Tasks Information

For detailed information on the available tasks, please refer to the documentation:
[RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html)

## Usage

### Data Collection
Users are encouraged to perform data collection themselves due to the high configurability and diversity of task and embodiment setups.  See the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for more details.

### 1. Task Running and Data Collection
```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Task Config
See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

## Policy Baselines
### Policies Supported
*   DP, ACT, DP3, RDT, PI0
*   TinyVLA, DexVLA (Contributed by Media Group)
*   Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, DexVLA, OpenVLA-OFT, SmolVLA, AVR, UniVLA

## Experiment & LeaderBoard
Coming Soon.

## Citations

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

## License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact

For questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).