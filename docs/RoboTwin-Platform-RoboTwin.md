# RoboTwin: The Premier Bimanual Robotic Manipulation Platform

RoboTwin is a cutting-edge platform designed for research and development in bimanual robotic manipulation, offering a scalable data generator, benchmark, and a strong domain randomization environment.  Explore the capabilities and features of RoboTwin on the [original RoboTwin GitHub repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Advanced Bimanual Robotic Manipulation:** Focuses on tasks requiring two-arm coordination.
*   **Scalable Data Generation:**  Provides tools to create diverse and large datasets for training and evaluation.
*   **Domain Randomization:** Implements strong domain randomization techniques for robust performance across varied environments.
*   **Benchmark Capabilities:**  Offers a comprehensive benchmark for evaluating and comparing different robotic manipulation algorithms.
*   **Multiple Versions:** Includes access to RoboTwin 1.0, 2.0, and early versions.
*   **Active Development:** The platform is constantly being updated with the latest research and features.

## RoboTwin Versions and Resources:

*   **RoboTwin 2.0 (Latest):**  
    *   [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Community](https://robotwin-platform.github.io/doc/community/index.html)
    *   Includes advancements in scalability, domain randomization, and benchmark tasks.
    *   Released July 2025
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:**
    *   Technical Report: [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)
*   **RoboTwin 1.0:**
    *   Accepted to <i style="color: red; display: inline;"><b>CVPR 2025 (Highlight)</b></i>: [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **Early Version:**
    *   Accepted to <i style="color: red; display: inline;"><b>ECCV Workshop 2024 (Best Paper Award)</b></i>: [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Quick Start:

### Installation:

Please see the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for installation instructions. The process typically takes about 20 minutes.

### Tasks and Usage:

*   For detailed information on tasks, refer to the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).
*   Explore the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for comprehensive usage instructions.
*   Utilize the provided example script: `bash collect_data.sh ${task_name} ${task_config} ${gpu_id}`

### Policy Baselines:
*   DP, ACT, DP3, RDT, PI0
*   TinyVLA, DexVLA
*   LLaVA-VLA
*   [Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Contributions and Community:

*   For questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).
*   Contributions are welcome!

## Citations:
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}

@InProceedings{Mu_2025_CVPR,
    author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
    title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27649-27660}
}

@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}

@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## License

Released under the MIT license.  See [LICENSE](./LICENSE) for details.