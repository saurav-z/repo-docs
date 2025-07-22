# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab is an open-source, GPU-accelerated framework that simplifies robotics research workflows, offering powerful simulation capabilities for reinforcement learning, imitation learning, and motion planning.**  Discover how to leverage the power of NVIDIA Isaac Sim for your robotics projects.  **(See the original repository on GitHub: [IsaacLab](https://github.com/isaac-sim/IsaacLab))**

![Isaac Lab](docs/source/_static/isaaclab.jpg)

**Key Features:**

*   **High-Fidelity Simulation:** Leverage NVIDIA Isaac Sim for accurate physics and sensor simulation, enabling sim-to-real transfer.
*   **GPU Acceleration:** Run complex simulations and computations faster with GPU acceleration, crucial for iterative processes like reinforcement learning.
*   **Extensive Robot Library:** Access a diverse collection of 16+ robot models, including manipulators, quadrupeds, and humanoids.
*   **Pre-built Environments:** Train your robots with more than 30 ready-to-use environments compatible with popular RL frameworks (RSL RL, SKRL, RL Games, Stable Baselines). Multi-agent RL is also supported.
*   **Sensor Simulation:** Accurate sensor models including RTX-based cameras, LIDAR, and contact sensors.
*   **Flexible Deployment:** Run simulations locally or distribute them across the cloud for large-scale experiments.

## Getting Started

Explore our comprehensive [documentation](https://isaac-sim.github.io/IsaacLab) for detailed tutorials and step-by-step guides, including:

*   [Installation Instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Examples](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

Ensure compatibility by referencing the correct Isaac Sim version for each Isaac Lab release:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

*   Note: The `feature/isaacsim_5_0` branch is actively updated and may contain breaking changes.
    *   It requires the [Isaac Sim 5.0 branch](https://github.com/isaac-sim/IsaacSim) built from source.

## Contributing

We welcome community contributions! Please review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details on submitting bug reports, feature requests, and code contributions.

## Show & Tell: Share Your Creations

Share your projects and inspire others in our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section.  Showcase your tutorials, learning content, and projects!

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues). For Isaac Sim-specific issues, refer to its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions, questions, and feature requests.
*   Report actionable items via Github [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with the NVIDIA Omniverse Community

Share your work and join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to collaborate and contribute to the Isaac Lab community.  Reach out to OmniverseCommunity@nvidia.com to explore opportunities to showcase your project.

## License

Isaac Lab is released under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension and its scripts are released under [Apache 2.0](LICENSE-mimic). Dependency licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is inspired by the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following in academic publications:

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