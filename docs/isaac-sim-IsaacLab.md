![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is an open-source, GPU-accelerated framework, built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html), designed to streamline robotics research workflows such as reinforcement learning, imitation learning, and motion planning. This makes Isaac Lab an excellent choice for researchers seeking to bridge the gap between simulation and real-world robotics applications.

**[Visit the original repository on GitHub](https://github.com/isaac-sim/IsaacLab)**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab empowers robotics researchers with a comprehensive suite of tools and features:

*   **Diverse Robot Models:** Access a library of 16 commonly available robot models, including manipulators, quadrupeds, and humanoids.
*   **Extensive Environment Support:** Train your models in over 30 ready-to-use environments, compatible with popular reinforcement learning frameworks (RSL RL, SKRL, RL Games, Stable Baselines, and multi-agent RL).
*   **Advanced Physics Engine:** Utilize a robust physics engine to simulate rigid bodies, articulated systems, and deformable objects.
*   **Realistic Sensor Simulation:** Leverage accurate sensor simulation capabilities, including RTX-based cameras, LiDAR, and contact sensors.
*   **GPU Acceleration:** Benefit from GPU acceleration for faster simulation and computation, crucial for iterative processes like reinforcement learning.
*   **Flexible Deployment:** Run simulations locally or distribute them across the cloud for large-scale experiments.

## Getting Started

Explore the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab) for detailed tutorials and guides:

*   [Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab requires specific versions of Isaac Sim, with the `main` branch supporting Isaac Sim 4.5.  The `feature/isaacsim_5_0` branch supports Isaac Sim 5.0 and may have some breaking changes.

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

## Contributing

The Isaac Lab team welcomes community contributions. Find details on the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell: Share Your Work

Share your projects and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) area.

## Troubleshooting

See the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For Isaac Sim-specific issues, see its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for ideas, questions, and feature requests.
*   [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports, documentation issues, new features, or general updates.

## Connect with the NVIDIA Omniverse Community

For broader project sharing, contact the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com, or join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE), and the `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). Dependency and asset licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Please cite the [Orbit](https://isaac-orbit.github.io/) framework in academic publications:

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