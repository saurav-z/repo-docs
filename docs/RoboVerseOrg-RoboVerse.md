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

## RoboVerse: Your Gateway to Scalable Robot Learning 🤖

RoboVerse is a comprehensive platform, dataset, and benchmark designed to accelerate research in scalable and generalizable robot learning.  Check out the [original repository](https://github.com/RoboVerseOrg/RoboVerse).

### Key Features

*   **Unified Platform:** Provides a centralized environment for robotics research.
*   **Extensive Dataset:** Includes a rich dataset for training and evaluating robot learning models.
*   **Robust Benchmarks:** Offers standardized benchmarks to measure and compare performance.
*   **Flexible Simulation:** Integrates with various simulation frameworks for diverse testing scenarios.
*   **Community-Driven:** Actively welcomes contributions and feedback from the open-source community.

### What's New

*   **[2025-04-10]**: RoboVerse accepted by RSS 2025!
*   **[2025-04-03]**: Code Released! The codebase is continuously updated.

### Getting Started

Dive into RoboVerse with the help of these resources:

*   **Detailed Documentation:** [Documentation](https://roboverse.wiki/metasim/#)
*   **Quick Start Tutorials:** [Quick Start Tutorials](https://roboverse.wiki/metasim/get_started/quick_start/0_static_scene)

### Contribute

We encourage your contributions to help make RoboVerse even better! See our [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on how to contribute.

### Feature Requests & Wish List

Have an idea for a new feature or improvement?  Share it in the [GitHub Discussions Wish List](https://github.com/RoboVerseOrg/RoboVerse/discussions/categories/wish-list). Upvote your favorite requests to help us prioritize!

### License and Acknowledgments

RoboVerse is released under the [Apache License 2.0](LICENSE).

This project leverages the following simulation frameworks, renderers, and libraries:

*   [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (built on [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html))
*   [Isaac Gym](https://developer.nvidia.com/isaac-gym)
*   [MuJoCo](https://github.com/google-deepmind/mujoco)
*   [SAPIEN](https://github.com/haosulab/SAPIEN)
*   [PyBullet](https://github.com/bulletphysics/bullet3)
*   [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)
*   [cuRobo](https://github.com/NVlabs/curobo)
*   [PyRep](https://github.com/stepjam/PyRep) (built on [CoppeliaSim](https://www.coppeliarobotics.com/))
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

The licenses for assets used in RoboVerse will be added soon. Please contact us if you have any questions.

### Citation

If you find RoboVerse valuable, please cite our work:

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