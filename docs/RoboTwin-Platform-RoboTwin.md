# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

[**RoboTwin**](https://github.com/RoboTwin-Platform/RoboTwin) is a cutting-edge platform designed to push the boundaries of dual-arm robotic manipulation, offering a comprehensive suite of tools for research and development.

## Key Features:

*   **Scalable Data Generation:** RoboTwin 2.0 features a powerful data generator to create diverse and realistic training datasets.
*   **Strong Domain Randomization:** Built-in domain randomization techniques enhance the robustness and generalizability of models.
*   **Comprehensive Benchmark:** Evaluate your bimanual robotic manipulation algorithms with our extensive benchmark suite.
*   **Pre-collected Large-scale Dataset:** Access a vast, pre-collected dataset of over 100,000 trajectories to kickstart your research. ([RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset))
*   **Policy Baselines:** Explore and deploy your policies with the support of a broad range of baselines: [DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html), [OpenVLA-oft](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html), [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html), [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html)

## Latest Updates & Announcements:

*   **2025/08/06:** RoboTwin 2.0 Leaderboard launched! ([leaderboard website](https://robotwin-platform.github.io/leaderboard)).
*   **2025/07/23:** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/01:** Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop released. [[arXiv](https://arxiv.org/abs/2506.23351)]
*   **2025/06/21:** RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **2025/04/11:** RoboTwin is selected as *CVPR Highlight paper*!
*   **2025/02/27:** RoboTwin is accepted to *CVPR 2025* !
*   **2024/09/30:** RoboTwin (Early Version) received *the Best Paper Award at the ECCV Workshop*!
*   **2024/09/20:** Officially released RoboTwin.

## Installation:

Follow the detailed installation instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation takes approximately 20 minutes.

## Tasks Information:

Explore the diverse range of tasks available in RoboTwin by visiting the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

## Usage:

*   **Document:** [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for comprehensive guidance.
*   **Data Collection:**  Easily collect data using the provided scripts. For example, run `bash collect_data.sh ${task_name} ${task_config} ${gpu_id}`.
*   **Modify Task Config:** Customize your experiments by modifying task configurations - see [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Experiment & Leaderboard:

Test your models and compare results on the RoboTwin Leaderboard: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard). Explore single-task fine-tuning, visual robustness, language diversity robustness, and cross-embodiment performance.

## Citations:

If you utilize RoboTwin in your research, please cite the following publications:

**RoboTwin 2.0:** A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin:** Dual-Arm Robot Benchmark with Generative Digital Twins, accepted to <i style="color: red; display: inline;"><b>CVPR 2025 (Highlight)</b></i>
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

Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin:** Dual-Arm Robot Benchmark with Generative Digital Twins (early version), accepted to <i style="color: red; display: inline;"><b>ECCV Workshop 2024 (Best Paper Award)</b></i>
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgements:

Software Support: D-Robotics, Hardware Support: AgileX Robotics, AIGC Support: Deemos.

Contact [Tianxing Chen](https://tianxingchen.github.io) for any questions or suggestions.

## License:

This project is released under the MIT license. See the [LICENSE](./LICENSE) file for details.