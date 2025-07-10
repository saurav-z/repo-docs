![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim, designed to streamline robotics research in areas like reinforcement learning and motion planning.  ([See the original repo](https://github.com/isaac-sim/IsaacLab) for more details).

---

## Key Features of Isaac Lab

Isaac Lab offers a powerful suite of features to advance your robotics projects:

*   **Diverse Robot Models:** Access a library of 16 commonly available robot models, including manipulators, quadrupeds, and humanoids.
*   **Extensive Environments:**  Train your models in over 30 pre-built environments, compatible with popular RL frameworks such as RSL RL, SKRL, and Stable Baselines.  MARL (multi-agent reinforcement learning) is also supported.
*   **Realistic Physics Simulation:**  Leverage accurate physics simulation with rigid bodies, articulated systems, and deformable objects.
*   **Advanced Sensor Simulation:**  Utilize a range of sensor options including RGB/depth/segmentation cameras, IMUs, and contact sensors for robust data collection.
*   **GPU Acceleration:**  Run complex simulations and computations faster thanks to GPU acceleration, enabling rapid iteration in RL and data-intensive tasks.
*   **Flexible Deployment:** Run Isaac Lab locally or in the cloud for flexibility.

## Getting Started

Explore these resources to quickly get up and running with Isaac Lab:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Overview](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab has specific version requirements for Isaac Sim.  Here's a summary:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

*Note: The `feature/isaacsim_5_0` branch is under active development and requires the Isaac Sim 5.0 branch built from source.*

## Contribute to Isaac Lab

Your contributions are valuable!  We encourage bug reports, feature requests, and code contributions.  See our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details.

## Showcase Your Work

Share your projects and tutorials in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions area.  Inspire others and contribute to the community!

## Troubleshooting and Support

*   Refer to the [Troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section.
*   Report issues on [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues).
*   Discuss ideas and ask questions on [GitHub Discussions](https://github.com/isaac-sim/IsaacLab/discussions).

## Connect with the NVIDIA Omniverse Community

Learn more about sharing projects and resources by contacting the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com.

Join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is based on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following paper in your academic publications:

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