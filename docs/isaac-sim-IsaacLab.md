![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is a powerful, open-source framework built on NVIDIA Isaac Sim, designed to revolutionize robotics research by unifying workflows for reinforcement learning, imitation learning, and motion planning, enabling faster and more accurate sim-to-real transfer.  Explore the capabilities of Isaac Lab on the [original repository](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab offers a comprehensive suite of features for advanced robotics simulation and learning:

*   **Diverse Robot Models:** Includes 16 pre-built robot models, including manipulators, quadrupeds, and humanoids.
*   **Ready-to-Train Environments:** Provides over 30 environments for training with popular RL frameworks (RSL RL, SKRL, RL Games, Stable Baselines), including support for multi-agent RL.
*   **Advanced Physics Simulation:** Supports rigid bodies, articulated systems, and deformable objects for realistic interactions.
*   **Realistic Sensor Simulation:** Includes RTX-based cameras, LIDAR, IMU, contact sensors, and ray casters for accurate data acquisition.

## Getting Started

Start your robotics simulation journey with the comprehensive resources available:

*   **[Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation):** Steps to set up Isaac Lab on your local machine.
*   **[Reinforcement Learning Examples](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html):** Learn how to implement RL algorithms.
*   **[Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html):** Step-by-step guides to get you started.
*   **[Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html):** Explore the available environments for training your robots.

## Isaac Sim Version Dependency

Isaac Lab is compatible with specific versions of Isaac Sim:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.1`                      | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

*Note: The `feature/isaacsim_5_0` branch is actively updated and may contain breaking changes.  It requires the [Isaac Sim 5.0 branch](https://github.com/isaac-sim/IsaacSim) built from source.*

## Contributing

We welcome community contributions!  Review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn how to contribute bug reports, feature requests, or code.

## Show & Tell: Share Your Robotics Projects

Showcase your projects and inspire the community in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions, questions, and feature requests.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for reporting bugs, documentation errors, new features, and updates.

## Connect with the NVIDIA Omniverse Community

Share your projects and resources by contacting the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com or joining the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its standalone scripts are released under [Apache 2.0](LICENSE-mimic).

## Acknowledgement

Isaac Lab originated from the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in your academic work:

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