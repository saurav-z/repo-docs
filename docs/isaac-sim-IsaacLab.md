# Isaac Lab: Accelerate Your Robotics Research with NVIDIA Isaac Sim

**Isaac Lab** is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim, streamlining robotics research workflows for reinforcement learning, imitation learning, and motion planning; explore [the original repo here](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Built upon the powerful NVIDIA Isaac Sim platform, Isaac Lab provides a robust and efficient environment for robotics researchers. It accelerates simulations with GPU power, providing the speed and accuracy needed for advanced robotics projects.

## Key Features of Isaac Lab:

*   **Comprehensive Robot Library:** Access a diverse range of robot models, including manipulators, quadrupeds, and humanoids, with 16 commonly available models.
*   **Ready-to-Train Environments:** Explore over 30 pre-built environments for reinforcement learning, supporting frameworks like RSL RL, SKRL, RL Games, and Stable Baselines, with multi-agent reinforcement learning support.
*   **Advanced Physics Simulation:** Simulate rigid bodies, articulated systems, and deformable objects with high accuracy.
*   **Realistic Sensor Simulation:** Utilize RTX-based cameras, LIDAR, IMU, contact sensors, and ray casters for accurate data acquisition.
*   **Flexible Deployment:** Run simulations locally or leverage cloud-based infrastructure for large-scale experiments.

## Getting Started

Explore the [official documentation](https://isaac-sim.github.io/IsaacLab) for detailed tutorials, guides, and installation instructions:

*   [Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab is compatible with specific versions of Isaac Sim. The following table details version dependencies:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

## Contribute and Collaborate

We welcome community contributions through bug reports, feature requests, and code contributions.  See our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell: Share Your Projects

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section to inspire and collaborate with the community.

## Troubleshooting

See the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common fixes, or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussions, questions, and new feature requests.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for tracking executable work with a clear scope and deliverable.

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com or joining the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and related scripts are licensed under [Apache 2.0](LICENSE-mimic). License files for dependencies are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab originated from the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in academic publications:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}