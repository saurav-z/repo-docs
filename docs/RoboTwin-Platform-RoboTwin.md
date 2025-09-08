# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

RoboTwin is a comprehensive platform for bimanual robotic manipulation research, featuring a scalable data generator, a challenging benchmark, and strong domain randomization. Explore the RoboTwin platform on [GitHub](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Comprehensive Benchmark:** Evaluate and compare algorithms on a diverse set of bimanual manipulation tasks.
*   **Scalable Data Generation:** Easily generate large-scale, realistic datasets for training and evaluating models.
*   **Strong Domain Randomization:** Enhance the robustness of your models with advanced domain randomization techniques.
*   **Pre-collected Datasets:** Access a large pre-collected dataset on Hugging Face.
*   **Variety of Policies Supported:** Offers support for multiple policies like DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA.
*   **Active Community:** Engage with a community of researchers and developers.
*   **Regular Updates:** Stay up-to-date with the latest advancements, including new tasks, baselines, and features.

## What's New

*   **RoboTwin 2.0:** Released with significant improvements in scalability, domain randomization, and benchmark challenges ([Webpage](https://robotwin-platform.github.io/)).
*   **CVPR 2025 Highlight Paper:** RoboTwin version 1.0 was accepted to CVPR 2025 as a highlight paper.
*   **CVPR 2025 Challenge:** Participate in the RoboTwin Dual-Arm Collaboration Challenge at the CVPR 2025 MEIS Workshop.
*   **ECCV 2024 Best Paper Award:** RoboTwin (Early Version) was awarded the Best Paper at the ECCV Workshop 2024.
*   **Leaderboard:** Evaluate your models and compare your results on the official RoboTwin leaderboard ([Leaderboard](https://robotwin-platform.github.io/leaderboard)).

## Getting Started

### Installation

Follow the detailed installation instructions available in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation should take about 20 minutes.

### Tasks

Explore the variety of tasks available on the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

![RoboTwin Tasks](https://github.com/RoboTwin-Platform/RoboTwin/raw/main/assets/files/50_tasks.gif)

### Usage

*   **Document:** Refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for detailed usage information.

*   **Data Collection:** Collect your own data or use the pre-collected trajectories available in the [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset) on Hugging Face.

    ![Domain Randomization](https://github.com/RoboTwin-Platform/RoboTwin/raw/main/assets/files/domain_randomization.png)

*   **Task Running and Data Collection:**

    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

*   **Modify Task Config:** Consult the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) to customize task configurations.

### Policy Baselines

RoboTwin supports a range of policy baselines to get you started:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   OpenVLA-oft
*   TinyVLA
*   DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))
*   **Deploy Your Policy:** Learn how to deploy your own policies using the [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

## Experiment & Leaderboard

The RoboTwin platform facilitates research into the following topics:

1.  Single-task fine-tuning.
2.  Visual robustness.
3.  Language diversity robustness (language condition).
4.  Multi-task capability.
5.  Cross-embodiment performance.

The complete leaderboard and settings can be found at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Access the pre-collected large-scale dataset on [Hugging Face](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you use RoboTwin, please cite the following:

**RoboTwin 2.0:**
```bibtex
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin:**
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

**RoboTwin Dual-Arm Collaboration Challenge:**
```bibtex
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (Early Version):**
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

For questions and suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is released under the MIT license. See [LICENSE](./LICENSE) for details.