# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Isaac Lab is an open-source framework built on NVIDIA Isaac Sim, revolutionizing robotics research by providing a fast, accurate, and GPU-accelerated simulation environment for reinforcement learning, imitation learning, and motion planning.** ([See the original repository](https://github.com/isaac-sim/IsaacLab))

[![Isaac Lab](docs/source/_static/isaaclab.jpg)](https://github.com/isaac-sim/IsaacLab)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab streamlines robotics research by integrating high-fidelity physics and sensor simulation, perfect for sim-to-real transfer.  It leverages the power of NVIDIA Isaac Sim to provide researchers with a powerful platform for developing and testing robotic solutions.

## Key Features of Isaac Lab

*   **Extensive Robot Library:** Explore and experiment with a diverse range of robots, including manipulators, quadrupeds, and humanoids (16+ models).
*   **Ready-to-Train Environments:** Utilize over 30 pre-built environments compatible with popular RL frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and multi-agent RL.
*   **Advanced Physics Simulation:** Simulate realistic interactions with rigid bodies, articulated systems, and deformable objects.
*   **Realistic Sensor Simulation:** Access RTX-based cameras, LIDAR, and contact sensors for accurate environmental perception.
*   **GPU Acceleration:** Benefit from accelerated simulations and faster computation for iterative processes like reinforcement learning.
*   **Flexible Deployment:** Run simulations locally or distribute them across the cloud for large-scale projects.

## Getting Started

Get up and running with Isaac Lab using the comprehensive documentation:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning with Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab is designed to work with specific versions of Isaac Sim.

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

The `feature/isaacsim_5_0` branch is under active development and may contain breaking changes.  It currently requires the [Isaac Sim 5.0 branch](https://github.com/isaac-sim/IsaacSim) built from source.

## Contribute to Isaac Lab

We welcome community contributions!  Review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn how to contribute code, report bugs, and suggest features.

## Showcase Your Work

Share your projects and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section, or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For Isaac Sim-related issues, consult the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Discuss ideas, ask questions, and request features in the GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions).
*   Report bugs, documentation issues, and track new features in GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com. Engage with other developers on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding scripts are released under [Apache 2.0](LICENSE-mimic).  Dependency and asset licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is built upon the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in your academic work:

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