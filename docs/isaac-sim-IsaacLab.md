![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with NVIDIA Isaac Sim

**Isaac Lab** is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim, designed to streamline robotics research by providing a unified environment for reinforcement learning, imitation learning, and motion planning, with fast, accurate physics and sensor simulation.  ([View the original repository](https://github.com/isaac-sim/IsaacLab))

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features for Robotics Simulation and Learning

Isaac Lab offers a powerful suite of tools and environments for robotics research:

*   **Extensive Robot Library**:  Includes 16 pre-built robot models, from manipulators to humanoids.
*   **Ready-to-Use Environments**:  Over 30 pre-configured environments compatible with popular RL frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and multi-agent RL.
*   **Advanced Physics Simulation**: Support for rigid bodies, articulated systems, and deformable objects.
*   **Realistic Sensor Simulation**: Provides RGB/depth/segmentation cameras, IMUs, contact sensors, and ray casters.

## Getting Started with Isaac Lab

Explore the comprehensive resources to get started with Isaac Lab:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Step-by-Step Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab has specific dependencies on NVIDIA Isaac Sim versions.

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.1`                      | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

**Note:** The `feature/isaacsim_5_0` branch is under active development and may have breaking changes. For Isaac Sim 5.0 compatibility, build the [Isaac Sim 5.0 branch](https://github.com/isaac-sim/IsaacSim) from source.

## Contribute to Isaac Lab

Your contributions are welcome! Help improve Isaac Lab through bug reports, feature requests, or code submissions.  See the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Share Your Projects (Show & Tell)

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the `Discussions` to inspire others.

## Troubleshooting and Support

*   Check the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common issues.
*   Submit issues [here](https://github.com/isaac-sim/IsaacLab/issues).
*   For Isaac Sim related issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).
*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions and feature requests.
*   Use Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports, documentation fixes, and new features.

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com or joining the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is released under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension is released under the [Apache 2.0](LICENSE-mimic) license.

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