# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

RoboTwin is a state-of-the-art platform for bimanual robotic manipulation, offering a scalable data generator, benchmark, and challenges to advance the field.  Explore cutting-edge research and contribute to the future of robotics.  [Visit the original repository](https://github.com/RoboTwin-Platform/RoboTwin) for more details.

## Key Features

*   **Scalable Data Generation:** Generate diverse and realistic datasets with ease for training and evaluating robotic manipulation skills.
*   **Strong Domain Randomization:**  Robustly train and evaluate algorithms in simulation with advanced domain randomization techniques.
*   **Comprehensive Benchmark:** Evaluate your robotic manipulation algorithms across a wide range of tasks and metrics.
*   **Active Community:** Participate in challenges, collaborate with researchers, and access community resources.
*   **Multiple Versions & Branches:** Access multiple versions, including a 2.0 release, and explore different development branches for specific features and challenges.
*   **Leaderboard:** Compete and track your performance on the RoboTwin leaderboard.

## RoboTwin Versions & Resources

*   **RoboTwin 2.0:**  [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper (arXiv)](https://arxiv.org/abs/2506.18088) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:** [Technical Report (arXiv)](https://arxiv.org/abs/2506.23351)
*   **RoboTwin 1.0:**  Accepted to CVPR 2025 (Highlight) - [Paper (arXiv)](https://arxiv.org/abs/2504.13059)
*   **RoboTwin (Early Version):** Accepted to ECCV Workshop 2024 (Best Paper Award) - [Paper (arXiv)](https://arxiv.org/abs/2409.02920)

## Installation

Installation takes approximately 20 minutes.  Detailed instructions are available in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).

## Tasks & Usage

Explore a wide array of manipulation tasks. More details can be found in [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

**Data Collection:**

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

**Task Configuration:**

Refer to the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for detailed configuration options.

## Policy Baselines

RoboTwin supports a variety of policy baselines, including:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   TinyVLA
*   DexVLA
*   LLaVA-VLA

Deploy your own policies by following this [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

## Experiment & Leaderboard

We encourage users to explore RoboTwin for:

*   Single-task fine-tuning
*   Visual robustness
*   Language diversity robustness
*   Multi-task capabilities
*   Cross-embodiment performance

Find the complete leaderboard and settings at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Citations

Please cite the following papers if you use RoboTwin:

```bibtex
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

## Acknowledgements

Software Support: D-Robotics, Hardware Support: AgileX Robotics, AIGC Support: Deemos

## License

This project is released under the MIT license. See [LICENSE](./LICENSE) for more details.

## Contact

Contact [Tianxing Chen](https://tianxingchen.github.io) for questions or suggestions.
```
Key improvements and SEO considerations:

*   **Clear Title & Hook:**  The initial title and one-sentence hook establish what RoboTwin is and its value.
*   **Keyword Integration:** Keywords like "bimanual robotic manipulation," "benchmark," "data generation," "domain randomization," and "robotics" are naturally incorporated.
*   **Headings & Structure:**  Organized with clear headings and subheadings for readability and SEO benefits.
*   **Bulleted Key Features:**  Highlights core functionalities, making it easy for users to understand the benefits.
*   **Concise Language:**  Uses straightforward language to convey information efficiently.
*   **Internal Linking:** Links to the various versions, resources and the original repo encourage user exploration.
*   **Call to Action:** Encourages users to explore the platform and contribute.
*   **Proper Formatting:** Uses markdown for better readability and SEO.
*   **Complete Information:** Keeps all the relevant information from the original README.
*   **SEO-Friendly Citations:** Includes bibtex for easy copy-pasting.
*   **Contact & License Information:**  Maintains important details.