![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is a cutting-edge, open-source framework built on NVIDIA Isaac Sim, designed to streamline robotics research with fast, accurate, and GPU-accelerated simulation.  [Explore the Isaac Lab project on GitHub](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab empowers robotics researchers with:

*   **Rich Robot Library:** Access to a diverse selection of over 16 pre-built robot models, including manipulators, quadrupeds, and humanoids.
*   **Extensive Environment Support:**  Train robots in over 30 ready-to-use environments, compatible with popular reinforcement learning frameworks like RSL RL, SKRL, RL Games, and Stable Baselines. Supports multi-agent reinforcement learning.
*   **Realistic Physics Simulation:**  Simulate rigid bodies, articulated systems, and deformable objects with high accuracy.
*   **Advanced Sensor Simulation:** Utilize realistic RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, and ray casters.
*   **GPU Acceleration:** Run complex simulations and computations at accelerated speeds for faster iteration in reinforcement learning and data-intensive tasks.
*   **Flexible Deployment:** Run simulations locally or in the cloud for scalability.

## Getting Started

Get up and running quickly with our comprehensive documentation:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Overview](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab is designed to work seamlessly with specific versions of Isaac Sim.

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

Note: The `feature/isaacsim_5_0` branch contains active updates and may have breaking changes.

## Contribute to Isaac Lab

We welcome contributions!  Please review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for information on how to contribute to the project.

## Share Your Work: Show & Tell

Share your projects and tutorials in our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) forum to inspire the community.

## Troubleshooting

Find solutions to common issues in our [troubleshooting guide](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues). For Isaac Sim-specific issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Discuss ideas, ask questions, and request features in [Discussions](https://github.com/isaac-sim/IsaacLab/discussions).
*   Report bugs, documentation issues, or new features in [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the NVIDIA Omniverse Community by reaching out to OmniverseCommunity@nvidia.com.  Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) for collaboration and community engagement.

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE), with the `isaaclab_mimic` extension licensed under [Apache 2.0](LICENSE-mimic).

## Acknowledgement

Isaac Lab is built on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in academic publications:

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