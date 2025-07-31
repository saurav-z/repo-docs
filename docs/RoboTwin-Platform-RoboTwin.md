# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

[RoboTwin's GitHub Repository](https://github.com/RoboTwin-Platform/RoboTwin)

RoboTwin is a cutting-edge platform designed to benchmark and advance the field of bimanual robotic manipulation, providing a scalable data generator, robust domain randomization, and a variety of tasks to test robotic capabilities.

**Key Features:**

*   **Scalable Data Generation:** Easily create diverse and large datasets for training and evaluation.
*   **Robust Domain Randomization:**  Enhance the generalization of your models with advanced domain randomization techniques.
*   **Comprehensive Benchmarks:**  Evaluate your algorithms on a variety of challenging bimanual robotic manipulation tasks.
*   **Multiple Task Support:** Includes 50+ tasks across manipulation, and bimanual interaction.
*   **Open-Source and Accessible:**  Available under the MIT license, enabling community contributions and improvements.
*   **Active Development:**  Regular updates with new features, tasks, and baselines.
*   **Latest Version 2.0**: Latest release with enhanced features, see [Webpage](https://robotwin-platform.github.io/) for more details.

**What's New in RoboTwin 2.0?**

*   Enhanced domain randomization techniques
*   Refined Task Setting
*   Fix DP3 evaluation code error
*   Updated endpose control mode
*   New Baselines: [DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   New Baselines (Contributed by Media Group): [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html)
*   New Baselines (Contributed by IRPN Lab, HKUST(GZ)): [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html)
*   RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   RoboTwin is seclected as <i>CVPR Highlight paper</i>!
*   RoboTwin is accepted to <i>CVPR 2025</i> !
*   RoboTwin (Early Version) received <i>the Best Paper Award at the ECCV Workshop</i>!

**Getting Started**

1.  **Installation:** Follow the detailed installation instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes about 20 minutes.
2.  **Tasks:** Explore the various tasks available in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).
3.  **Usage:**  Refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for data collection, task configuration, and more.
4.  **Run example task:**
```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```
5.  **Policy Baselines:** Experiment with different control policies like DP, ACT, DP3, and RDT and LLaVA-VLA, deploy your own policy using the [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)
6.  **Data Collection:** Run your own data collection or download [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

**Citations**

If you use RoboTwin in your research, please cite the following papers:

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

**License:**

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

**Acknowledgments**

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos
*   **Code Style:** `find . -name "*.py" -exec sh -c 'echo "Processing: {}"; yapf -i --style='"'"'{based_on_style: pep8, column_limit: 120}'"'"' {}' \;`

**Contact:**

For questions and suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).