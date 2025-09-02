# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

RoboTwin is a comprehensive platform designed to advance the field of bimanual robotic manipulation, offering a scalable data generator, robust domain randomization, and a challenging benchmark.  [Explore the RoboTwin Repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Quickly generate diverse datasets for training and evaluating robotic manipulation models.
*   **Robust Domain Randomization:**  Utilize strong domain randomization techniques to enhance the robustness of your models in real-world scenarios.
*   **Comprehensive Benchmark:** Evaluate your models against state-of-the-art baselines on a variety of challenging bimanual tasks.
*   **Open-Source & Community-Driven:** Benefit from the open-source nature of the project, fostering collaboration and continuous improvement.
*   **Leaderboard:** Track your progress and compare your results with others on the RoboTwin leaderboard.
*   **Multiple Versions & Challenges:** Access different versions (1.0, 2.0) and participate in challenges like the CVPR 2025 MEIS Workshop.

## What's New

*   **RoboTwin 2.0:** The latest version introduces improvements in data generation, domain randomization, and offers an even more challenging benchmark.
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:** Participate in the challenge to benchmark and improve your bimanual robotic manipulation algorithms.
*   **RoboTwin 1.0:**  The foundational benchmark for bimanual robotic manipulation, providing a robust and versatile platform for research and development.

## Resources

*   **Webpage:** [https://robotwin-platform.github.io/](https://robotwin-platform.github.io/)
*   **Documentation:** [https://robotwin-platform.github.io/doc/](https://robotwin-platform.github.io/doc/)
*   **Paper:** [https://arxiv.org/abs/2506.18088](https://arxiv.org/abs/2506.18088)
*   **Community:** [https://robotwin-platform.github.io/doc/community/index.html](https://robotwin-platform.github.io/doc/community/index.html)
*   **Leaderboard:** [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard)

## Installation

Follow the detailed installation instructions in the RoboTwin 2.0 Document: [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

## Tasks

Explore a diverse set of bimanual manipulation tasks:

*   For details, see: [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html)

```
<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>
```

## Usage

### Document
> Please Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for more details.

### Data Collection
We provide over 100,000 pre-collected trajectories as part of the open-source release [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
However, we strongly recommend users to perform data collection themselves due to the high configurability and diversity of task and embodiment setups.

```
<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">
```

### 1. Task Running and Data Collection
Running the following command will first search for a random seed for the target collection quantity, and then replay the seed to collect data.

```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Modify Task Config
☝️ See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

## Policy Baselines

### Policies Support

[DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html), [OpenVLA-oft](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html)

[TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)

[LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html) (Contributed by IRPN Lab, HKUST(GZ))

Deploy Your Policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, SmolVLA, AVR, UniVLA

## Experiment & Leaderboard

> We recommend that the RoboTwin Platform can be used to explore the following topics: 
> 1. single - task fine - tuning capability
> 2. visual robustness
> 3. language diversity robustness (language condition)
> 4. multi-tasks capability
> 5. cross-embodiment performance

The full leaderboard and setting can be found in: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Please refer to [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you use RoboTwin in your research, please cite the following:

```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

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

```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

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

This project is released under the MIT License.  See the [LICENSE](./LICENSE) file for more details.

## Contact

For questions and suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).