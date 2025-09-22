# RoboTwin: A Comprehensive Benchmark for Bimanual Robotic Manipulation

**RoboTwin** is a cutting-edge platform designed to advance research in bimanual robotic manipulation, offering a scalable data generator, benchmark, and robust domain randomization capabilities; check out the original repo [here](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Generate diverse and realistic datasets for training and evaluating bimanual robotic manipulation algorithms.
*   **Strong Domain Randomization:** Enhance the robustness of your models through advanced domain randomization techniques.
*   **Comprehensive Benchmark:** Evaluate and compare your algorithms on a wide range of challenging bimanual manipulation tasks.
*   **Open-Source Dataset:** Access over 100,000 pre-collected trajectories [here](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
*   **Multiple Policy Baselines:**  Includes support for various policy baselines like DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA.

## Latest Updates

*   **RoboTwin 2.0** (Under Review 2025) - [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop** - [Technical Report](https://arxiv.org/abs/2506.23351)
*   **RoboTwin 1.0** (CVPR 2025 Highlight) - [Paper](https://arxiv.org/pdf/2504.13059)
*   **RoboTwin (Early Version)** (ECCV Workshop 2024 Best Paper Award) - [Paper](https://arxiv.org/pdf/2409.02920)

## Installation

See [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for detailed installation instructions.

## Tasks

Explore a diverse range of bimanual manipulation tasks.  See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Document

Refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for comprehensive usage details.

### Data Collection

Collect data by using the following command:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Task Configuration

Customize task parameters. See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines

RoboTwin supports multiple policy baselines:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   OpenVLA-oft
*   TinyVLA
*   DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))

Deploy your policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

Explore the RoboTwin platform for:

1.  Single-task fine-tuning
2.  Visual robustness
3.  Language diversity robustness (language condition)
4.  Multi-task capability
5.  Cross-embodiment performance

The full leaderboard can be found at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Dataset

Access the pre-collected large-scale dataset on Hugging Face: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you use RoboTwin in your research, please cite the relevant papers:

*   **RoboTwin 2.0**:

    ```
    @article{chen2025robotwin,
      title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
      author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
      journal={arXiv preprint arXiv:2506.18088},
      year={2025}
    }
    ```

*   **RoboTwin**:

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

*   **Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop**:

    ```
    @article{chen2025benchmarking,
      title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
      author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
      journal={arXiv preprint arXiv:2506.23351},
      year={2025}
    }
    ```

*   **RoboTwin (Early Version)**:

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

This project is released under the MIT License.  See the [LICENSE](./LICENSE) file for details.

## Contact

For any questions or suggestions, contact [Tianxing Chen](https://tianxingchen.github.io).