# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

**RoboTwin** is a comprehensive platform providing a scalable data generator, benchmark, and challenging tasks for advancing research in bimanual robotic manipulation. [Visit the RoboTwin GitHub Repository](https://github.com/RoboTwin-Platform/RoboTwin) to get started.

## Key Features

*   **Comprehensive Benchmark:** Includes diverse tasks and evaluation metrics.
*   **Scalable Data Generation:** Generates realistic data with domain randomization for robust training.
*   **Strong Domain Randomization:** Enhances the generalizability of trained models.
*   **Pre-collected Datasets:** Access a large-scale, pre-collected dataset on Hugging Face.
*   **Policy Baselines:**  Offers support for various state-of-the-art policies like DP, ACT, DP3, RDT, PI0, and OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA.
*   **Active Development:**  Regular updates, including new tasks, baselines, and dataset expansions.
*   **Community Driven:** Join the community to discuss and share your insights.

## Recent Updates
*   **[2025/08/28]**  Updated RoboTwin 2.0 Paper [[PDF]](https://arxiv.org/pdf/2506.18088).
*   **[2025/08/25]**  Fixed ACT deployment code and updated the [[leaderboard]](https://robotwin-platform.github.io/leaderboard).
*   **[2025/08/06]**  Released RoboTwin 2.0 Leaderboard: [[leaderboard website]](https://robotwin-platform.github.io/leaderboard).
*   **[2025/07/23]**  RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **[2025/07/19]**  Fixed DP3 evaluation code error. 
*   **[2025/07/09]**  Updated endpose control mode.
*   **[2025/07/08]**  Uploaded Challenge-Cup-2025 Branch.
*   **[2025/07/02]**  Fixed Piper Wrist Bug.
*   **[2025/07/01]**  Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **[2025/06/21]**  Released RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **[2025/04/11]**  RoboTwin is seclected as CVPR Highlight paper!
*   **[2025/02/27]**  RoboTwin is accepted to CVPR 2025 ! 
*   **[2024/09/30]**  RoboTwin (Early Version) received the Best Paper Award  at the ECCV Workshop!
*   **[2024/09/20]**  Officially released RoboTwin.

## Installation

Follow the detailed installation instructions found in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes around 20 minutes.

## Tasks

Explore a variety of bimanual manipulation tasks. See the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more information.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Data Collection

1.  **Run the following command:**
    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```
2.  **Modify Task Configuration:** Refer to the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

### Available Policies

*   [DP](https://robotwin-platform.github.io/doc/usage/DP.html)
*   [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html)
*   [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html)
*   [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html)
*   [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   [OpenVLA-oft](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html)
*   [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)
*   [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)
*   [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html) (Contributed by IRPN Lab, HKUST(GZ))
*   **Deploy Your Policy:** [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiments & Leaderboard

Explore the RoboTwin platform for:

1.  Single-task fine-tuning.
2.  Visual robustness.
3.  Language diversity robustness.
4.  Multi-task capabilities.
5.  Cross-embodiment performance.

View the full leaderboard and settings at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Dataset

Access the pre-collected RoboTwin 2.0 Dataset on Hugging Face: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

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

For questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.