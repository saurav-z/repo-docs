![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim, designed to revolutionize robotics research workflows by simplifying and unifying reinforcement learning, imitation learning, and motion planning.  [Explore the original repository](https://github.com/isaac-sim/IsaacLab)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features of Isaac Lab

Isaac Lab empowers robotics researchers with a comprehensive suite of tools, including:

*   **Extensive Robot Library:** Includes 16 pre-built robot models, from manipulators to humanoids, ready for simulation.
*   **Rich Environment Support:** Offers over 30 ready-to-train environments compatible with popular reinforcement learning frameworks. Supports single and multi-agent reinforcement learning.
*   **Advanced Physics Simulation:** Provides realistic simulation of rigid bodies, articulated systems, and deformable objects.
*   **High-Fidelity Sensor Simulation:** Offers accurate sensor simulation with RTX-based cameras, LIDAR, IMU, and contact sensors.
*   **GPU Acceleration:** Enables faster simulation and computation, ideal for iterative processes like reinforcement learning.
*   **Flexible Deployment:**  Runs locally or in the cloud, facilitating large-scale deployments.

## Getting Started

Get up and running with Isaac Lab quickly using our detailed documentation:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Overview](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab is designed to work with specific versions of NVIDIA Isaac Sim.  Here's a compatibility overview:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

**Important:**  The `feature/isaacsim_5_0` branch is under active development and requires [Isaac Sim 5.0](https://github.com/isaac-sim/IsaacSim) built from source.  See the branch's README for details.

## Contribute

We encourage community contributions!  Review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn how to contribute code, report bugs, or suggest features.

## Showcase Your Work

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section. Inspire others and contribute to the community!

## Troubleshooting and Support

*   Check the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common issues.
*   Submit issues via [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues).
*   Discuss ideas and ask questions in [GitHub Discussions](https://github.com/isaac-sim/IsaacLab/discussions).
*   For Isaac Sim-related issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Connect with the NVIDIA Omniverse Community

Share your work and collaborate with other developers on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse). Reach out to the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to explore opportunities to spotlight your work.

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic). Dependencies' and assets' licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

This project is built upon the [Orbit](https://isaac-orbit.github.io/) framework.  Please cite it in academic publications:

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