# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is an open-source framework built on NVIDIA Isaac Sim, designed to streamline robotics research workflows using GPU-accelerated physics and sensor simulation, perfect for reinforcement learning, imitation learning, and sim-to-real transfer.  Learn more and contribute at the [original repo](https://github.com/isaac-sim/IsaacLab).

[![Isaac Lab](docs/source/_static/isaaclab.jpg)]()

## Key Features

*   **Extensive Robot Library:** Access a wide array of 16 commonly used robot models, including manipulators, quadrupeds, and humanoids.
*   **Diverse Environment Support:** Train your models in over 30 pre-built environments, compatible with popular RL frameworks such as RSL RL, SKRL, and Stable Baselines. Multi-agent RL is also supported.
*   **Advanced Physics Simulation:** Leverage rigid bodies, articulated systems, and deformable objects for realistic simulations.
*   **Realistic Sensor Simulation:** Utilize RTX-based cameras, LiDAR, IMUs, and contact sensors for accurate and detailed data acquisition.
*   **GPU Acceleration:** Benefit from GPU-accelerated computation for faster simulations and quicker iteration cycles.

## Getting Started

Find comprehensive tutorials, installation guides, and environment overviews in the Isaac Lab [documentation](https://isaac-sim.github.io/IsaacLab).

## Isaac Sim Compatibility

Isaac Lab is designed to work seamlessly with specific versions of NVIDIA Isaac Sim. Please refer to the following table to ensure compatibility:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

## Contribute

Contribute to the project through bug reports, feature requests, or code contributions. See our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell

Share your tutorials, projects, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) area of the Discussions section and inspire others!

## Troubleshooting

Consult the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common fixes or submit an issue on [GitHub](https://github.com/isaac-sim/IsaacLab/issues).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general questions and discussions.
*   Submit GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for executable work with a defined scope.

## Connect with the NVIDIA Omniverse Community

Share your projects and resources by contacting the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com, and join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) for collaboration.

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension is licensed under [Apache 2.0](LICENSE-mimic).

## Acknowledgements

Isaac Lab is developed from the [Orbit](https://isaac-orbit.github.io/) framework.  Please cite the following in academic publications:

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