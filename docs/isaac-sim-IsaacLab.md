# Isaac Lab: Accelerate Robotics Research with NVIDIA Isaac Sim

**Isaac Lab is a powerful, open-source framework designed to revolutionize robotics research by unifying and simplifying your workflows using the speed and accuracy of NVIDIA Isaac Sim.** Explore the original repository on [GitHub](https://github.com/isaac-sim/IsaacLab).

[![Isaac Lab](docs/source/_static/isaaclab.jpg)](https://github.com/isaac-sim/IsaacLab)

## Key Features

*   **GPU-Accelerated Simulation:** Leverage the power of NVIDIA GPUs for fast and accurate physics and sensor simulation, ideal for iterative processes like reinforcement learning.
*   **Extensive Robot Library:** Access a diverse collection of robots, including manipulators, quadrupeds, and humanoids, with 16 commonly available models.
*   **Pre-built Environments:** Train your robots with over 30 ready-to-use environments, compatible with popular reinforcement learning frameworks like RSL RL, SKRL, and Stable Baselines. Includes support for multi-agent reinforcement learning.
*   **Realistic Physics and Sensors:** Simulate rigid bodies, articulated systems, deformable objects, and accurate sensor models including RGB/depth/segmentation cameras, IMUs, and contact sensors.
*   **Flexible Deployment:** Run simulations locally or distribute them across the cloud for large-scale robotics research.

## Getting Started

Dive into robotics simulation with our comprehensive documentation:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning with Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

Isaac Lab is built on Isaac Sim and requires specific versions. Refer to the table for compatibility:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

**Note:** The `feature/isaacsim_5_0` branch is under active development and may have breaking changes.

## Contribute & Share

We welcome contributions! Learn how to contribute by reviewing our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

Share your projects and learn from others in our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) discussion area.

## Troubleshooting & Support

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or submit an issue [here](https://github.com/isaac-sim/IsaacLab/issues).

For Isaac Sim specific issues, refer to the [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) and [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions and questions.
*   Report specific issues, bugs, and feature requests via GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with the NVIDIA Omniverse Community

Join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect with developers and share your work.

## License

Isaac Lab is released under the [BSD-3 License](LICENSE).

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. Please cite:

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