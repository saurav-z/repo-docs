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

# RoboVerse: The Unified Platform for Scalable Robot Learning

**RoboVerse is a comprehensive platform, dataset, and benchmark designed to advance scalable and generalizable robot learning.**

[View the original repository on GitHub](https://github.com/RoboVerseOrg/RoboVerse)

## Key Features

*   **Unified Platform:** Integrates multiple simulation frameworks and datasets for a versatile robot learning environment.
*   **Rich Dataset:** Leverages a diverse range of datasets including RLBench, ManiSkill, and more (see below for a complete list).
*   **Benchmarking Capabilities:** Enables standardized evaluation and comparison of robot learning algorithms.
*   **Open Source:** Encourages community contributions and collaboration.

## What's New

*   **[2025-04-10]** RoboVerse accepted by RSS 2025!
*   **[2025-04-03]** Code released! The codebase is continuously evolving. Contributions are welcome.

## Getting Started

For detailed instructions on how to get started with RoboVerse, please refer to the official [documentation](https://roboverse.wiki/metasim/#) and [tutorials](https://roboverse.wiki/metasim/get_started/quick_start/0_static_scene).

## Contribute

We welcome contributions from the community!  Please review the [CONTRIBUTING.md](./CONTRIBUTING.md) file for guidelines on how to contribute.

## Feature Requests & Wish List

Have ideas for new features or improvements?  Share them in the "Wish List" section of our [GitHub Discussions](https://github.com/RoboVerseOrg/RoboVerse/discussions/categories/wish-list). Upvote the requests you find most relevant.

## License and Acknowledgments

RoboVerse is licensed under the Apache License 2.0.

This project utilizes the following simulation frameworks, renderers, and libraries:

*   Isaac Lab ([Isaac Lab](https://github.com/isaac-sim/IsaacLab))
*   Isaac Gym ([Isaac Gym](https://developer.nvidia.com/isaac-gym))
*   MuJoCo ([MuJoCo](https://github.com/google-deepmind/mujoco))
*   SAPIEN ([SAPIEN](https://github.com/haosulab/SAPIEN))
*   PyBullet ([PyBullet](https://github.com/bulletphysics/bullet3))
*   Genesis ([Genesis](https://github.com/Genesis-Embodied-AI/Genesis))
*   cuRobo ([cuRobo](https://github.com/NVlabs/curobo))
*   PyRep ([PyRep](https://github.com/stepjam/PyRep))
*   Blender ([Blender](https://www.blender.org/))

RoboVerse also integrates data from the following projects:

*   RLBench ([RLBench](https://github.com/stepjam/RLBench))
*   ManiSkill ([ManiSkill](https://github.com/haosulab/ManiSkill))
*   LIBERO ([LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO))
*   Meta-World ([Meta-World](https://github.com/Farama-Foundation/Metaworld))
*   robosuite ([robosuite](https://github.com/ARISE-Initiative/robosuite))
*   GraspNet ([GraspNet](https://graspnet.net/))
*   ARNOLD ([ARNOLD](https://arnold-benchmark.github.io/))
*   GAPartNet ([GAPartNet](https://github.com/PKU-EPIC/GAPartNet))
*   GAPartManip ([GAPartManip](https://arxiv.org/abs/2411.18276))
*   UniDoorManip ([UniDoorManip](https://github.com/sectionZ6/UniDoorManip))
*   SimplerEnv ([SimplerEnv](https://github.com/simpler-env/SimplerEnv))
*   RLAfford ([RLAfford](https://github.com/hyperplane-lab/RLAfford))
*   Open6DOR ([Open6DOR](https://github.com/Selina2023/Open6DOR))
*   CALVIN ([CALVIN](https://github.com/mees/calvin))
*   GarmentLab ([GarmentLab](https://github.com/GarmentLab/GarmentLab))
*   Matterport3D ([Matterport3D](https://github.com/niessner/Matterport))
*   VLN-CE ([VLN-CE](https://github.com/jacobkrantz/VLN-CE))
*   vMaterials ([vMaterials](https://developer.nvidia.com/vmaterials))
*   HumanoidBench ([HumanoidBench](https://github.com/carlosferrazza/humanoid-bench))

## Citation

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