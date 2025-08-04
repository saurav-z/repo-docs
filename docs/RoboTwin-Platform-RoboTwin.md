# RoboTwin: Your Comprehensive Platform for Bimanual Robotic Manipulation

RoboTwin provides a powerful platform for research and development in bimanual robotic manipulation, offering a scalable data generator, a robust benchmark, and a collection of pre-trained baselines.  [Explore the RoboTwin GitHub Repository](https://github.com/RoboTwin-Platform/RoboTwin) for more information!

## Key Features

*   **Scalable Data Generation:** Easily create diverse datasets for training and evaluating bimanual robotic manipulation algorithms.
*   **Strong Domain Randomization:**  Enhances the robustness of your models by simulating real-world variations.
*   **Comprehensive Benchmark:** Provides a standardized evaluation framework with a variety of tasks and metrics.
*   **Pre-trained Policy Baselines:**  Get started quickly with implementations of popular algorithms like DP, ACT, DP3, RDT, PI0, TinyVLA, DexVLA, and LLaVA-VLA.
*   **Community and Resources:** Access detailed documentation, a community forum, and pre-collected datasets.
*   **Actively Developed:**  RoboTwin is constantly updated with new features, tasks, and improvements.

## What's New?

*   **RoboTwin 2.0:**  The latest version of RoboTwin (released June 2025) offers significant advancements, including enhanced domain randomization, new tasks, and improved documentation.
*   **RoboTwin Dual-Arm Collaboration Challenge:**  Participate in the CVPR 2025 MEIS Workshop challenge, focusing on generalizable bimanual manipulation.

## Key Highlights

*   **CVPR 2025 Highlight Paper:** RoboTwin has been recognized as a highlight paper at CVPR 2025.
*   **ECCV Workshop 2024 Best Paper Award:** The early version of RoboTwin was awarded the Best Paper at the ECCV Workshop 2024.

## Getting Started

### Installation

Detailed installation instructions are available in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes about 20 minutes.

### Tasks Information

Refer to [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details on the available tasks.

### Usage

For comprehensive usage instructions, please see the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

#### Data Collection

RoboTwin allows you to collect your own data to maximize configurability and diversity.

Run the following command to collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

#### Task Configuration

See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for task configuration details.

### Policy Baselines

*   **Supported Policies:** [DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html), [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) , [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html)
*   **Deploy Your Policy:** [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

### Experiment & Leaderboard

Coming Soon.

## Citations

Please cite the following papers if you use RoboTwin:

*   **RoboTwin 2.0:**

    ```
    @article{chen2025robotwin,
      title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
      author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
      journal={arXiv preprint arXiv:2506.18088},
      year={2025}
    }
    ```

*   **RoboTwin:**

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

*   **RoboTwin Dual-Arm Collaboration Challenge:**

    ```
    @article{chen2025benchmarking,
      title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
      author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
      journal={arXiv preprint arXiv:2506.23351},
      year={2025}
    }
    ```

*   **RoboTwin (Early Version):**

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

## License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).