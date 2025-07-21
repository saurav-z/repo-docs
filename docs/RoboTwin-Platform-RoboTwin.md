# RoboTwin: The Ultimate Bimanual Robotic Manipulation Platform

**RoboTwin** is an advanced benchmark and data generator for bimanual robotic manipulation, offering a robust platform for research and development. Check out the [original repository](https://github.com/RoboTwin-Platform/RoboTwin) for more details.

## Key Features:

*   **Scalable Data Generation:** Generate diverse datasets for training and evaluating robotic manipulation skills.
*   **Strong Domain Randomization:** Enhance the robustness of your models with extensive domain randomization techniques.
*   **Comprehensive Benchmark:** Evaluate and compare different bimanual robotic manipulation algorithms.
*   **Pre-collected Trajectories:** Utilize over 100,000 pre-collected trajectories from the [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset) to get started quickly.
*   **Multiple Versions:** Access various versions of RoboTwin to explore different advancements and challenges.
*   **Community Support:** Engage with the community and find all resources on the [community page](https://robotwin-platform.github.io/doc/community/index.html)
*   **Multiple supported algorithms:**
    *   DP
    *   ACT
    *   DP3
    *   RDT
    *   PI0
    *   TinyVLA
    *   DexVLA

## Key Resources:

*   **Latest Version (RoboTwin 2.0):** [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088)
*   **RoboTwin Dual-Arm Collaboration Challenge:** [Technical Report](https://arxiv.org/pdf/2506.23351)
*   **RoboTwin 1.0:** [CVPR 2025 Highlight Paper](https://arxiv.org/pdf/2504.13059)
*   **Early Version:** [ECCV Workshop 2024 Best Paper Award](https://arxiv.org/pdf/2409.02920)

## Installation

Follow the detailed instructions in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for a smooth installation process (approximately 20 minutes).

## Tasks and Usage

Explore a wide array of tasks with RoboTwin, see the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).  Learn how to collect data and configure tasks in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/index.html).

## Policy Baselines

RoboTwin supports various policy baselines to kickstart your research, including:

*   [DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)
*   Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Citation

If you use RoboTwin in your research, please cite the following papers:

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

Special thanks to D-Robotics for Software Support, AgileX Robotics for Hardware Support, and Deemos for AIGC Support.

## License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for more details.