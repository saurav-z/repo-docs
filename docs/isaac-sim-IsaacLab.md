![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Isaac Lab** is a powerful, open-source framework built on NVIDIA Isaac Sim, designed to streamline and accelerate robotics research in areas like reinforcement learning, imitation learning, and motion planning.  ([Explore the original repository](https://github.com/isaac-sim/IsaacLab))

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab leverages the power of GPU acceleration for fast and accurate physics and sensor simulation, making it ideal for sim-to-real transfer in robotics. It provides a unified platform for robotics researchers.

## Key Features

*   **Diverse Robot Models:** Includes a wide array of robots, such as manipulators, quadrupeds, and humanoids, with 16 commonly available models.
*   **Extensive Environments:** Offers ready-to-train implementations of over 30 environments compatible with popular reinforcement learning frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and supports multi-agent reinforcement learning.
*   **Advanced Physics Simulation:** Supports rigid bodies, articulated systems, and deformable objects for realistic simulations.
*   **Comprehensive Sensor Suite:** Provides accurate sensor simulation with RGB/depth/segmentation cameras, IMUs, contact sensors, and ray casters.

## Getting Started

Get up and running with Isaac Lab quickly with our comprehensive documentation: [https://isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab)

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Examples](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Environment Overview](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab is compatible with specific versions of NVIDIA Isaac Sim. Ensure you use the correct Isaac Sim version for your Isaac Lab release:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.1`                      | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

*Note:* The `feature/isaacsim_5_0` branch is actively updated and requires [Isaac Sim 5.0](https://github.com/isaac-sim/IsaacSim) built from source.

## Contributing

We welcome community contributions!  Please refer to our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Showcase Your Work

Share your projects and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section. Inspire others and contribute to the robotics community.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For Isaac Sim specific issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions, questions, and feature requests.
*   Use Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) for tracking specific bugs, documentation issues, or features.

## Connect with the NVIDIA Omniverse Community

Share your projects with the NVIDIA Omniverse Community at OmniverseCommunity@nvidia.com or join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension is licensed under [Apache 2.0](LICENSE-mimic). See the [`docs/licenses`](docs/licenses) directory for dependency licenses.

## Acknowledgement

Isaac Lab builds upon the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in your academic publications:

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
```