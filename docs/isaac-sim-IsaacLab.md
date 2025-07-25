![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Unleash the Power of Robotics Simulation with NVIDIA Isaac Sim

**Isaac Lab** is an open-source, GPU-accelerated robotics framework built on NVIDIA Isaac Sim, designed to accelerate your robotics research workflows. Learn more and contribute at the [Isaac Lab GitHub repository](https://github.com/isaac-sim/IsaacLab).

---

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


## What is Isaac Lab?

Isaac Lab is a powerful, open-source framework built upon NVIDIA Isaac Sim, designed to streamline robotics research. It empowers researchers and developers to accelerate their robotics projects through a unified and GPU-accelerated platform, ideal for tasks like reinforcement learning, imitation learning, and motion planning.  It leverages fast and accurate physics and sensor simulation for efficient sim-to-real transfer.

## Key Features:

*   **Comprehensive Robot Models:** Access a diverse collection of 16+ pre-built robot models, including manipulators, quadrupeds, and humanoids.
*   **Rich Environment Library:** Train your models with over 30 ready-to-use environments, supporting popular reinforcement learning frameworks like RSL RL, SKRL, RL Games, and Stable Baselines, as well as multi-agent reinforcement learning.
*   **Realistic Physics Simulation:** Experience accurate rigid body, articulated systems, and deformable object physics.
*   **Advanced Sensor Simulation:** Utilize RTX-based cameras, LIDAR, contact sensors, and more for realistic sensor data.

## Getting Started

Jumpstart your robotics research with our comprehensive documentation:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Resources](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab requires specific versions of Isaac Sim. Here's the compatibility matrix:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

*   The `feature/isaacsim_5_0` branch is actively updated and may contain breaking changes; it requires the [Isaac Sim 5.0 branch](https://github.com/isaac-sim/IsaacSim) built from source. Refer to its README for instructions.

## Contribute to Isaac Lab

We welcome community contributions through bug reports, feature requests, and code contributions. Explore our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn more.

## Showcase Your Work

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) area to inspire others and foster innovation in robotics and simulation.

## Troubleshooting

Find solutions to common issues in our [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).  For Isaac Sim-related issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions, questions, and feature requests.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for tracking specific, actionable tasks like bug fixes and new features.

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com. Join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to collaborate and contribute.

## License

Isaac Lab is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic). Dependency licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is built upon the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in academic publications:

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