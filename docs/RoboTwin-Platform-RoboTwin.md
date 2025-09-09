# RoboTwin: The Ultimate Bimanual Robotic Manipulation Platform

RoboTwin is a cutting-edge platform designed for research and development in bimanual robotic manipulation, offering a scalable data generator, comprehensive benchmark, and strong domain randomization capabilities.  Explore the platform at the original repository: [https://github.com/RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Generate vast amounts of diverse and realistic data for training robotic manipulation models.
*   **Robust Domain Randomization:**  Overcome the reality gap through advanced domain randomization techniques.
*   **Comprehensive Benchmark:** Evaluate and compare different bimanual robotic manipulation algorithms.
*   **Diverse Tasks:** Supports a wide range of manipulation tasks for thorough evaluation.
*   **Pre-collected Dataset:** Access a large-scale, pre-collected dataset for immediate use.
*   **Community Support:** Engage with a growing community of researchers and developers.
*   **Leaderboard:** Track and compare performance on a public leaderboard.

## What's New

*   **RoboTwin 2.0 (Latest Version):** Offers significant advancements in scalability, data generation, and benchmark capabilities. ([Webpage](https://robotwin-platform.github.io/), [Document](https://robotwin-platform.github.io/doc), [Paper](https://arxiv.org/abs/2506.18088))
*   **RoboTwin Dual-Arm Collaboration Challenge:**  Participate in the challenge and push the boundaries of bimanual robotic manipulation. ([Technical Report](https://arxiv.org/abs/2506.23351))

## Latest Updates

*   **2025/08/28**: Updated RoboTwin 2.0 paper ([PDF](https://arxiv.org/pdf/2506.18088)).
*   **2025/08/25**: Fixed ACT deployment code and updated the [leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **2025/08/06**: Released RoboTwin 2.0 Leaderboard.
*   **(See the full list in the original README for more updates!)**

## Installation

Follow the detailed installation instructions in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes around 20 minutes.

## Tasks & Usage

RoboTwin supports numerous bimanual manipulation tasks.  For detailed information on tasks and usage, please refer to:

*   **Tasks Information:** [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html)
*   **Usage:** [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html)
*   **Data Collection:**  Scripts are provided to facilitate data collection. Example command: `bash collect_data.sh ${task_name} ${task_config} ${gpu_id}`
*   **Task Configurations:**  Refer to the [Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) to modify task settings.

## Policy Baselines

RoboTwin supports various policy baselines for robotic manipulation:

*   DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, LLaVA-VLA
*   Deploy Your Policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

*   The [leaderboard](https://robotwin-platform.github.io/leaderboard) enables you to evaluate your policies on RoboTwin, with a focus on:
    *   Single-task fine-tuning
    *   Visual robustness
    *   Language diversity robustness
    *   Multi-task capability
    *   Cross-embodiment performance

## Pre-collected Large-scale Dataset

Explore the pre-collected dataset on Hugging Face: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you find RoboTwin valuable for your research, please cite the following papers:

*   **RoboTwin 2.0:**
    ```
    @article{chen2025robotwin,
      title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
      author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
      journal={arXiv preprint arXiv:2506.18088},
      year={2025}
    }
    ```
*   **RoboTwin (CVPR 2025 Highlight):**
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
*   **Benchmarking Generalizable Bimanual Manipulation:**
    ```
    @article{chen2025benchmarking,
      title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
      author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
      journal={arXiv preprint arXiv:2506.23351},
      year={2025}
    }
    ```
*   **RoboTwin (ECCV 2024 Best Paper Award):**
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

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.