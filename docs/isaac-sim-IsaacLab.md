![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim, revolutionizing robotics research by unifying and simplifying workflows for reinforcement learning, imitation learning, and motion planning. Explore the power of realistic physics and sensor simulation.  [Learn more on GitHub](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features of Isaac Lab

Isaac Lab offers a comprehensive suite of tools for efficient and effective robotics research:

*   **Diverse Robot Models:** Explore 16+ pre-built robot models including manipulators, quadrupeds, and humanoids.
*   **Extensive Environments:** Train your robot agents in 30+ ready-to-use environments, compatible with popular RL frameworks.
*   **Realistic Physics:** Leverage advanced physics simulations, including rigid bodies, articulated systems, and deformable objects.
*   **Advanced Sensors:** Utilize a variety of sensor options, including RTX-based cameras, LIDAR, and contact sensors for realistic data acquisition.
*   **GPU Acceleration:** Benefit from GPU-accelerated simulations for faster iteration in reinforcement learning and data-intensive tasks.
*   **Flexible Deployment:** Run simulations locally or in the cloud for scalable research.

## Getting Started

Jumpstart your robotics projects with our detailed documentation, tutorials, and guides.

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning with Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Compatibility

Isaac Lab is designed to work seamlessly with specific versions of NVIDIA Isaac Sim.  The following table details compatibility:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

**Note:** The `feature/isaacsim_5_0` branch is under active development and may have breaking changes.

## Contribute & Collaborate

We encourage community contributions!  Share your ideas, report bugs, request features, and contribute code.  See our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details.

## Showcase Your Work

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of our Discussions.

## Troubleshooting & Support

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section, or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues) on GitHub.

For Isaac Sim related issues, consult the [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general questions and feature requests.  Use Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) to track executable pieces of work.

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com.  Engage with other developers on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension is licensed under [Apache 2.0](LICENSE-mimic).  Dependency and asset licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following paper in academic publications:

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