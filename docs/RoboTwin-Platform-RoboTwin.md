# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

RoboTwin is a cutting-edge platform designed to advance research in bimanual robotic manipulation, offering a comprehensive benchmark for evaluating and comparing algorithms. Check out the original repository [here](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Scalable Data Generation:** Generate diverse and realistic datasets with ease.
*   **Strong Domain Randomization:**  Enhance the robustness of your algorithms through effective domain randomization techniques.
*   **Comprehensive Benchmark:**  Evaluate performance across a wide range of bimanual manipulation tasks.
*   **Open-Source and Accessible:**  Freely available with a permissive MIT license.
*   **Multiple Versions:** Includes RoboTwin 1.0, 2.0, and early versions.
*   **Community Driven:** Active development and contributions from researchers.

## What's New in RoboTwin 2.0?

**RoboTwin 2.0** is the latest version of the platform, offering significant improvements in scalability, domain randomization, and task variety. It features:

*   [Webpage](https://robotwin-platform.github.io/)
*   [Document](https://robotwin-platform.github.io/doc)
*   [Paper](https://arxiv.org/abs/2506.18088)

### Key Updates

*   **2025/07/23**: Outstanding Poster at ChinaSI 2025 (Ranking 1st)
*   **2025/07/19**: Fixed DP3 evaluation code error, update paper next week.
*   **2025/07/09**: Update endpose control mode
*   **2025/07/08**: Released Challenge Cup Branch
*   **2025/07/02**: Fixed Piper Wrist Bug - Redownload embodiment asset.
*   **2025/07/01**: Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop.
*   **2025/06/21**: Released RoboTwin 2.0 !
*   **2025/04/11**: Selected as CVPR Highlight paper.
*   **2025/02/27**: Accepted to CVPR 2025 !
*   **2024/09/30**: Received the Best Paper Award at the ECCV Workshop!
*   **2024/09/20**: Officially released RoboTwin

## Installation

Detailed installation instructions can be found in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). The installation process typically takes around 20 minutes.

## Tasks and Usage

RoboTwin supports a variety of bimanual manipulation tasks. 

### Data Collection

Data collection is highly configurable; it is recommended to perform it yourself for the best results.

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Task Configurations

Consult the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for configuration details.

## Policy Baselines

RoboTwin supports multiple policy baselines, including:

*   [DP](https://robotwin-platform.github.io/doc/usage/DP.html)
*   [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html)
*   [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html)
*   [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html)
*   [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)
*   [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)
*   [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html) (Contributed by IRPN Lab, HKUST(GZ))

Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

Experiment details and leaderboard information will be available soon.

## Citations

If you utilize RoboTwin in your research, please cite the relevant papers:

**RoboTwin 2.0:**
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin (CVPR 2025):**
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

**RoboTwin CVPR 2025 MEIS Workshop Challenge:**
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (ECCV Workshop 2024):**
```
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

This project is released under the [MIT License](LICENSE).