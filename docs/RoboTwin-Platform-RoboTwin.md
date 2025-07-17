<h1 align="center">
  <a href="https://github.com/RoboTwin-Platform/RoboTwin">
    <b>RoboTwin:  The Premier Bimanual Robotic Manipulation Platform</b>
  </a>
</h1>

RoboTwin is a cutting-edge platform for developing and benchmarking robust bimanual robotic manipulation, offering a comprehensive toolkit for researchers and developers.

## Key Features:

*   **Scalable Data Generation:** Generate diverse and realistic datasets with strong domain randomization.
*   **Comprehensive Benchmark:** Evaluate your algorithms across a variety of challenging bimanual tasks.
*   **Dual-Arm Collaboration Challenge:**  Participate in the RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop.
*   **Multiple Versions:** Access and compare the evolution of the platform with different versions (1.0, 2.0, and Early Versions).
*   **Extensive Documentation:** Detailed documentation, including installation instructions and usage guides.
*   **Policy Baselines:** Includes support for various policy baselines like DP, ACT, DP3, RDT, PI0, TinyVLA, and DexVLA.

## 🚀 Get Started

*   **Installation:** Follow the detailed instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes about 20 minutes.
*   **Tasks:** Explore the available tasks and configurations in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

## 📚 Overview

| Branch Name | Link |
|-------------|------|
| 2.0 Version Branch | [main](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) (latest) |
| 1.0 Version Branch | [1.0 Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0) |
| 1.0 Version Code Generation Branch | [1.0 Version GPT](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt) |
| Early Version Branch | [Early Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version) |
| 第十九届“挑战杯”人工智能专项赛分支 | [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) |
| CVPR 2025 Challenge Round 1 Branch | [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1) |
| CVPR 2025 Challenge Round 2 Branch | [CVPR-Challenge-2025-Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2) |

## 🐣 Update

*   **2025/07/09**, Endpose control mode added: [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)].
*   **2025/07/08**,  Challenge-Cup-2025 Branch updated: [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025)
*   **2025/07/02**, Piper Wrist Bug fixed. Please redownload embodiment asset.
*   **2025/07/01**, RoboTwin Dual-Arm Collaboration Challenge Technical Report released: [[arXiv](https://arxiv.org/abs/2506.23351)]!
*   **2025/06/21**, RoboTwin 2.0 released: [[Webpage](https://robotwin-platform.github.io/)]!
*   **2025/04/11**, RoboTwin is selected as <i>CVPR Highlight paper</i>!
*   **2025/02/27**, RoboTwin is accepted to <i>CVPR 2025</i>!
*   **2024/09/30**, RoboTwin (Early Version) received <i>the Best Paper Award at the ECCV Workshop</i>!
*   **2024/09/20**, Officially released RoboTwin.

## 🛠️ Installation

Detailed installation instructions can be found in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation time is about 20 minutes.

## 🤷‍♂️ Tasks Information

For more details, please consult [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## 🧑🏻‍💻 Usage

> Please Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for more details.

### Data Collection

We provide over 100,000 pre-collected trajectories as part of the open-source release [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset). However, we strongly recommend users to perform data collection themselves due to the high configurability and diversity of task and embodiment setups.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection

Run the following command to collect data. It first searches for a random seed for the target collection quantity, and then replays the seed to collect data.

```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Task Config

See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

## 🚴‍♂️ Policy Baselines

### Policies Support

[DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)

[TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)

Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, DexVLA, OpenVLA-OFT, SmolVLA, AVR, UniVLA

## 🏄‍♂️ Experiment & LeaderBoard

> We recommend that the RoboTwin Platform can be used to explore the following topics: 
> 1. single - task fine - tuning capability
> 2. visual robustness
> 3. language diversity robustness (language condition)
> 4. multi-tasks capability
> 5. cross-embodiment performance

Coming Soon.

## 👍 Citations

If you find our work useful, please consider citing:

**RoboTwin 2.0**:
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin**:
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

**Benchmarking Generalizable Bimanual Manipulation:**
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (early version)**:
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## 😺 Acknowledgement

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos

Code Style: `find . -name "*.py" -exec sh -c 'echo "Processing: {}"; yapf -i --style='"'"'{based_on_style: pep8, column_limit: 120}'"'"' {}' \;`

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## 🏷️ License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

**[Go to the original repository](https://github.com/RoboTwin-Platform/RoboTwin)**