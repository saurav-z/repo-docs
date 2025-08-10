# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

RoboTwin is a cutting-edge platform providing a comprehensive suite for research in bimanual robotic manipulation, with diverse tasks, domain randomization, and a dynamic leaderboard.  Explore the repository on [GitHub](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Diverse Tasks:** Explore over 50 challenging bimanual robotic manipulation tasks, from simple object manipulation to complex collaborative actions.
*   **Domain Randomization:** Robustly evaluate your algorithms with built-in domain randomization for enhanced generalization.
*   **Scalable Data Generation:** Generate high-quality, diverse datasets to train and evaluate your bimanual robotic manipulation models.
*   **Leaderboard:** Compare your models with state-of-the-art approaches on our dynamic leaderboard.
*   **Multiple Versions:** Access various versions including:
    *   **RoboTwin 2.0 (Latest):** Includes updated tasks, documentation, and the latest leaderboard.
    *   **RoboTwin 1.0:** The original benchmark version.
    *   **Early Version:** The initial release, recognized with the ECCV Workshop Best Paper Award.
*   **Community & Resources:**  Access a wealth of resources including:
    *   Webpage: [https://robotwin-platform.github.io/](https://robotwin-platform.github.io/)
    *   Documentation: [https://robotwin-platform.github.io/doc/](https://robotwin-platform.github.io/doc/)
    *   Paper: [https://arxiv.org/abs/2506.18088](https://arxiv.org/abs/2506.18088)
    *   Community: [https://robotwin-platform.github.io/doc/community/index.html](https://robotwin-platform.github.io/doc/community/index.html)
    *   Leaderboard: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard)

## Updates

*   **2025/08/06:** RoboTwin 2.0 Leaderboard released!
*   **2025/07/23:** RoboTwin 2.0 receives Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/01:** Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop released: [[arXiv](https://arxiv.org/abs/2506.23351)]
*   **2025/06/21:** RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **2025/04/11:** RoboTwin is selected as *CVPR Highlight paper*!
*   **2025/02/27:** RoboTwin is accepted to *CVPR 2025*!
*   **2024/09/30:** RoboTwin (Early Version) received the Best Paper Award at the ECCV Workshop!
*   **2024/09/20:** Officially released RoboTwin.

## Installation

Follow the detailed installation instructions in [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes about 20 minutes.

## Tasks Information

See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

Detailed usage instructions and examples are available in the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

### Data Collection

We provide pre-collected trajectories and also encourage you to collect your own data.

### 1. Task Running and Data Collection

Use the following command to collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Task Config

Explore various configuration options via [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines

### Policies Supported

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   TinyVLA
*   DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))

Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, DexVLA, OpenVLA-OFT, SmolVLA, AVR, UniVLA

## Experiment & Leaderboard

Explore:
1. single - task fine - tuning capability
2. visual robustness
3. language diversity robustness (language condition)
4. multi-tasks capability
5. cross-embodiment performance

The full leaderboard can be found at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Citations

If you use RoboTwin in your research, please cite the following:

**RoboTwin 2.0:**
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin:**
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

**Benchmarking Generalizable Bimanual Manipulation:**
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (Early Version):**
```
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

Contact [Tianxing Chen](https://tianxingchen.github.io) if you have any questions or suggestions.

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.