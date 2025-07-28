# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is a cutting-edge, open-source framework for robotics research, designed to simplify workflows for reinforcement learning, imitation learning, and motion planning, leveraging the power of NVIDIA Isaac Sim.

[![Isaac Lab](docs/source/_static/isaaclab.jpg)](https://github.com/isaac-sim/IsaacLab)

**Key Features:**

*   **GPU-Accelerated Simulation:** Run complex robotics simulations faster with NVIDIA's GPU acceleration, crucial for iterative processes.
*   **Diverse Robot Models:** Access a comprehensive library of 16 commonly available robot models, including manipulators, quadrupeds, and humanoids.
*   **Extensive Environments:** Train robots in over 30 pre-built environments, compatible with popular RL frameworks.
*   **Realistic Sensor Simulation:** Utilize RTX-based cameras, LIDAR, and contact sensors for accurate data acquisition.
*   **Flexible Deployment:** Run simulations locally or distribute them across the cloud for large-scale projects.
*   **Physics Engine:** Includes support for rigid bodies, articulated systems, and deformable objects.
*   **Comprehensive Tutorials and Documentation:**  Get started quickly with detailed tutorials and step-by-step guides.

## Why Use Isaac Lab?

Isaac Lab provides a unified platform for robotics research, enabling researchers to create realistic simulations, train robots in various environments, and transfer knowledge to real-world applications. Its GPU-accelerated physics and sensor simulation capabilities, combined with its wide range of features and support, make it an ideal choice for robotics research.

## Getting Started

Dive into the world of robotics simulation with Isaac Lab. Explore our documentation for detailed guides on:

*   [Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab is built on top of Isaac Sim and requires specific versions of Isaac Sim that are compatible with each release of Isaac Lab.  The table below outlines recent releases and their dependencies:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

## Contribute and Collaborate

We welcome contributions! Check out our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn how you can help improve Isaac Lab.

## Show and Tell

Share your projects and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) area.

## Troubleshooting and Support

*   Consult the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common issues.
*   Report issues or request features on [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues).
*   Discuss ideas and ask questions on [GitHub Discussions](https://github.com/isaac-sim/IsaacLab/discussions).

## Connect with the Community

Join the NVIDIA Omniverse community on [Discord](https://discord.com/invite/nvidiaomniverse) or contact OmniverseCommunity@nvidia.com.

## License

Isaac Lab is released under the [BSD-3 License](LICENSE).

The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic).

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. If you use Isaac Lab in your research, please cite the following:

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

[**Visit the original repository on GitHub**](https://github.com/isaac-sim/IsaacLab)