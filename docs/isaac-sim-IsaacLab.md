# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab is a cutting-edge, open-source framework for robotics research that leverages the power of NVIDIA Isaac Sim to provide fast, accurate, and GPU-accelerated simulation capabilities.** [Visit the original repository on GitHub](https://github.com/isaac-sim/IsaacLab).

[![Isaac Lab](docs/source/_static/isaaclab.jpg)](https://github.com/isaac-sim/IsaacLab)

## Key Features:

*   **High-Fidelity Simulation:** Built on NVIDIA Isaac Sim for realistic physics and sensor simulation.
*   **GPU Acceleration:** Run complex simulations faster, ideal for iterative processes like reinforcement learning.
*   **Diverse Robot Models:** Includes 16 pre-built robot models (manipulators, quadrupeds, humanoids) to start from.
*   **Ready-to-Train Environments:** Over 30 environments compatible with popular reinforcement learning frameworks.
*   **Comprehensive Sensor Simulation:** Includes RTX-based cameras, LIDAR, and contact sensors for accurate data.
*   **Flexible Deployment:** Run simulations locally or in the cloud.
*   **Extensive Physics Support:** Rigid bodies, articulated systems, and deformable objects.

## Getting Started

Explore our documentation for detailed tutorials and guides to kickstart your robotics projects.

*   [Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

## Contribute and Share

We welcome community contributions through bug reports, feature requests, and code contributions.

*   [Contribution Guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html)
*   Share your projects and tutorials in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section.

## Troubleshooting & Support

*   Refer to the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section.
*   Submit issues on [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues).
*   Discuss ideas and ask questions on [GitHub Discussions](https://github.com/isaac-sim/IsaacLab/discussions).
*   For Isaac Sim-related issues, see its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Connect with the NVIDIA Omniverse Community

Share your work with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com or join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic).

## Acknowledgement

Please cite [Orbit](https://isaac-orbit.github.io/) in your academic publications.

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