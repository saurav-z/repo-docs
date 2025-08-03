![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab is an open-source, GPU-accelerated robotics simulation framework built on NVIDIA Isaac Sim, streamlining robotics research workflows for reinforcement learning, imitation learning, and motion planning.**  Explore the [original repository on GitHub](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab offers a robust platform for robotics research, providing:

*   **Diverse Robot Models:** Access a wide range of robot models, including manipulators, quadrupeds, and humanoids.
*   **Ready-to-Train Environments:** Experiment with over 30 pre-built environments compatible with popular RL frameworks.
*   **Advanced Physics Simulation:**  Utilize rigid bodies, articulated systems, and deformable objects for realistic simulations.
*   **Realistic Sensor Simulation:** Benefit from RTX-based cameras, LIDAR, and contact sensors for accurate data acquisition.
*   **GPU-Accelerated Performance:**  Leverage GPU acceleration for faster simulation and computation, essential for iterative learning processes.
*   **Cloud-Ready Deployment:** Run simulations locally or distribute them across the cloud for scalability.

## Getting Started

Begin your robotics research journey with Isaac Lab!  Explore our comprehensive resources to get up and running:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Examples](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Interactive Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Compatibility

Isaac Lab is designed to work with specific versions of NVIDIA Isaac Sim.

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.1`                      | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

The `feature/isaacsim_5_0` branch is actively updated and may have breaking changes. Refer to its README for instructions.

## Contribute to Isaac Lab

Join the community!  We encourage contributions in the form of bug reports, feature requests, and code contributions. See the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Share Your Work

Showcase your projects and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section. Share tutorials, learning content, and exciting projects to foster innovation in robotics and simulation.

## Troubleshooting

Consult the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) guide or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues). For Isaac Sim-specific issues, check its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions, questions, and feature requests.
*   Report bugs, documentation issues, and other actionable items via GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com. Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect with developers and contribute to the ecosystem.

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and related scripts are released under [Apache 2.0](LICENSE-mimic).  License files for dependencies and assets can be found in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development is based on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in academic publications:

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