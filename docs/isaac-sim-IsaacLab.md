# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Isaac Lab is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim, revolutionizing robotics research by providing a unified platform for reinforcement learning, imitation learning, and motion planning.**  Check out the [original repository](https://github.com/isaac-sim/IsaacLab) for the full details.

[![Isaac Lab](docs/source/_static/isaaclab.jpg)](https://github.com/isaac-sim/IsaacLab)

Isaac Lab leverages the power of NVIDIA Isaac Sim to provide fast and accurate physics and sensor simulation, making it an ideal solution for sim-to-real transfer in robotics applications.  The framework's GPU acceleration allows for rapid iteration in reinforcement learning and other data-intensive tasks.

## Key Features

*   **Extensive Robot Library:**  Includes 16 commonly available robot models, from manipulators to humanoids, providing a wide range of options for your projects.
*   **Pre-built Environments:**  Offers over 30 ready-to-train environments compatible with popular reinforcement learning frameworks like RSL RL, SKRL, and Stable Baselines, with support for multi-agent reinforcement learning.
*   **Advanced Physics Simulation:**  Supports rigid bodies, articulated systems, and deformable objects for realistic environment interactions.
*   **Comprehensive Sensor Suite:**  Features RTX-based cameras, LIDAR, and contact sensors for accurate and detailed data capture, emulating real-world sensor behavior.

## Getting Started

The [documentation](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started with Isaac Lab.

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Overview](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

Isaac Lab requires specific versions of Isaac Sim:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

## Contribute

Contributions are welcome.  See the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details.

## Show & Tell

Share your projects and tutorials in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions.

## Troubleshooting

Check the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussions and questions.
*   Use Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) for executable pieces of work.

## Connect with the NVIDIA Omniverse Community

Share your projects with the community at OmniverseCommunity@nvidia.com.

## License

Isaac Lab is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and scripts are released under [Apache 2.0](LICENSE-mimic).

## Acknowledgement

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