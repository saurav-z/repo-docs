![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is a cutting-edge, open-source framework that revolutionizes robotics research by unifying and simplifying workflows for reinforcement learning, imitation learning, and motion planning.  Developed on NVIDIA Isaac Sim, it delivers fast, accurate physics and sensor simulation to bridge the gap between simulation and real-world robotic applications. [Learn more at the original repo](https://github.com/isaac-sim/IsaacLab)!

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features: Unlock the Power of Robotics Simulation

Isaac Lab provides a comprehensive toolkit for building and training robotic systems:

*   **Extensive Robot Library:** Includes a wide range of robot models, featuring 16 common robotic platforms such as manipulators, quadrupeds, and humanoids.
*   **Ready-to-Train Environments:** Offers over 30 pre-built environments compatible with popular reinforcement learning frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and supports multi-agent reinforcement learning.
*   **Advanced Physics Engine:** Simulate rigid bodies, articulated systems, and deformable objects for realistic interaction.
*   **Realistic Sensor Simulation:** Features a suite of sensors, including RGB/depth/segmentation cameras, IMUs, contact sensors, and ray casters, all utilizing RTX-based technology for accurate results.

## Getting Started with Isaac Lab

Start your robotics journey with our detailed documentation.

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning with Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Interactive Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Compatibility

Isaac Lab seamlessly integrates with NVIDIA Isaac Sim.  Ensure you're using a compatible version:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.1`                      | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

For the `feature/isaacsim_5_0` branch, refer to its README for instructions.

## Contribute and Collaborate

We welcome community contributions! Review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn how to contribute.

## Share Your Robotics Projects

Showcase your work in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section and contribute to the robotics community.

## Troubleshooting and Support

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section. For further assistance, please [submit an issue](https://github.com/isaac-sim/IsaacLab/issues) or engage in [Discussions](https://github.com/isaac-sim/IsaacLab/discussions). For Isaac Sim-related issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the community by contacting the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com and join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE), while the `isaaclab_mimic` extension is licensed under [Apache 2.0](LICENSE-mimic).  License files for dependencies and assets are located in the `docs/licenses` directory.

## Acknowledgement

If you use Isaac Lab in your academic work, please cite:

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