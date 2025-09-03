<p align="center">
  <img src="docs/source/_static/RoboVerse86.22.svg" width="50%" alt="RoboVerse">
</p>

![RoboVerse](docs/source/metasim/images/tea.jpg)

<p align="center">
  <a href="https://roboverseorg.github.io"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2504.18904"><img src="https://img.shields.io/badge/paper-preprint-red" alt="Paper"></a>
  <a href="https://roboverse.wiki"><img src="https://img.shields.io/badge/doc-page-orange" alt="Documentation"></a>
  <a href="https://github.com/RoboVerseOrg/RoboVerse/issues"><img src="https://img.shields.io/github/issues/RoboVerseOrg/RoboVerse?color=yellow" alt="Issues"></a>
  <a href="https://github.com/RoboVerseOrg/RoboVerse/discussions"><img src="https://img.shields.io/github/discussions/RoboVerseOrg/RoboVerse?color=blueviolet" alt="Discussions"></a>
  <a href="https://discord.gg/6e2CPVnAD3"><img src="https://img.shields.io/discord/1356345436927168552?logo=discord&color=blue" alt="Discord"></a>
  <a href="docs/source/_static/wechat.jpg"><img src="https://img.shields.io/badge/wechat-QR_code-green" alt="WeChat"></a>
</p>

## RoboVerse: A Unified Platform for Robot Learning (GitHub Repo)

**RoboVerse is a comprehensive platform offering a unified dataset, benchmark, and tools to advance the field of scalable and generalizable robot learning.**  [Explore the original RoboVerse repository on GitHub](https://github.com/RoboVerseOrg/RoboVerse).

### Key Features:

*   **Unified Platform:** A centralized resource for robot learning research.
*   **Comprehensive Dataset:** Provides a rich dataset to train and evaluate robotic agents.
*   **Benchmark Capabilities:** Enables standardized performance comparison across different learning approaches.
*   **Scalable and Generalizable Learning:** Focused on methods that can work across a variety of tasks and environments.
*   **Active Community:** Encourages open-source contributions and community involvement.

### Getting Started

Dive into the world of RoboVerse with the following resources:

*   **Documentation:** Access detailed information and guides on the [documentation page](https://roboverse.wiki/metasim/#).
*   **Tutorials:** Start quickly with our [tutorials](https://roboverse.wiki/metasim/get_started/quick_start/0_static_scene) that guide you through the initial steps.

### Contribute to RoboVerse

We welcome contributions from the community!  Find out how to contribute in [CONTRIBUTING.md](./CONTRIBUTING.md).

### Feature Requests & Wish List

Have ideas for new features, simulators, tasks, or workflows? Share them on our [GitHub Discussions](https://github.com/RoboVerseOrg/RoboVerse/discussions/categories/wish-list), where you can also upvote your favorite requests.

### License and Acknowledgments

RoboVerse is licensed under the Apache License 2.0.

The project leverages the following simulation frameworks, renderers, and libraries:

*   [Isaac Lab](https://github.com/isaac-sim/IsaacLab), built on top of [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
*   [Isaac Gym](https://developer.nvidia.com/isaac-gym)
*   [MuJoCo](https://github.com/google-deepmind/mujoco)
*   [SAPIEN](https://github.com/haosulab/SAPIEN)
*   [PyBullet](https://github.com/bulletphysics/bullet3)
*   [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)
*   [cuRobo](https://github.com/NVlabs/curobo)
*   [PyRep](https://github.com/stepjam/PyRep), built on top of [CoppeliaSim](https://www.coppeliarobotics.com/)
*   [Blender](https://www.blender.org/)

RoboVerse also integrates data from the following projects:

*   [RLBench](https://github.com/stepjam/RLBench)
*   [Maniskill](https://github.com/haosulab/ManiSkill)
*   [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
*   [Meta-World](https://github.com/Farama-Foundation/Metaworld)
*   [robosuite](https://github.com/ARISE-Initiative/robosuite)
*   [GraspNet](https://graspnet.net/)
*   [ARNOLD](https://arnold-benchmark.github.io/)
*   [GAPartNet](https://github.com/PKU-EPIC/GAPartNet)
*   [GAPartManip](https://arxiv.org/abs/2411.18276)
*   [UniDoorManip](https://github.com/sectionZ6/UniDoorManip)
*   [SimplerEnv](https://github.com/simpler-env/SimplerEnv)
*   [RLAfford](https://github.com/hyperplane-lab/RLAfford)
*   [Open6DOR](https://github.com/Selina2023/Open6DOR)
*   [CALVIN](https://github.com/mees/calvin)
*   [GarmentLab](https://github.com/GarmentLab/GarmentLab)
*   [Matterport3D](https://github.com/niessner/Matterport)
*   [VLN-CE](https://github.com/jacobkrantz/VLN-CE)
*   [vMaterials](https://developer.nvidia.com/vmaterials)
*   [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench)

*The licenses for the assets used in RoboVerse will be added soon.*

### Citation

If you use RoboVerse in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{geng2025roboverse,
      title={RoboVerse: Towards a Unified Platform, Dataset and Benchmark for Scalable and Generalizable Robot Learning},
      author={Haoran Geng and Feishi Wang and Songlin Wei and Yuyang Li and Bangjun Wang and Boshi An and Charlie Tianyue Cheng and Haozhe Lou and Peihao Li and Yen-Jen Wang and Yutong Liang and Dylan Goetting and Chaoyi Xu and Haozhe Chen and Yuxi Qian and Yiran Geng and Jiageng Mao and Weikang Wan and Mingtong Zhang and Jiangran Lyu and Siheng Zhao and Jiazhao Zhang and Jialiang Zhang and Chengyang Zhao and Haoran Lu and Yufei Ding and Ran Gong and Yuran Wang and Yuxuan Kuang and Ruihai Wu and Baoxiong Jia and Carlo Sferrazza and Hao Dong and Siyuan Huang and Yue Wang and Jitendra Malik and Pieter Abbeel},
      year={2025},
      eprint={2504.18904},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2504.18904},
}
```