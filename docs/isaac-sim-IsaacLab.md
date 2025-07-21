![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim, streamlining robotics research workflows for reinforcement learning, imitation learning, and motion planning.  [Explore the Isaac Lab repository here.](https://github.com/isaac-sim/IsaacLab)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


Isaac Lab provides a powerful and efficient environment for robotics research, leveraging the speed and accuracy of NVIDIA Isaac Sim.  It offers a rich set of features designed to simplify the development and evaluation of robotic systems.  Its GPU acceleration allows for faster simulation and computation, crucial for iterative processes like reinforcement learning.

## Key Features

*   **Robots:** Access a library of 16 pre-built robot models, including manipulators, quadrupeds, and humanoids.
*   **Environments:** Utilize over 30 ready-to-train environments compatible with popular RL frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and support for multi-agent reinforcement learning.
*   **Physics:** Simulate rigid bodies, articulated systems, and deformable objects for realistic interactions.
*   **Sensors:** Emulate realistic sensor data with support for RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, and ray casters.
*   **Scalability:** Run simulations locally or in the cloud for flexible deployment.

## Getting Started

Jumpstart your robotics research with our comprehensive documentation.

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning with Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab is designed to work with specific versions of Isaac Sim. Stay updated on compatibility:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

## Contribute

We welcome contributions to improve Isaac Lab!  Find details in the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Showcase Your Work

Share your projects and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section.  For Isaac Sim specific problems, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Discuss ideas and ask questions in [Discussions](https://github.com/isaac-sim/IsaacLab/discussions).
*   Report bugs, request features, and track work in [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with the NVIDIA Omniverse Community

Share your robotics projects and connect with the community by contacting the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com.  Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to collaborate and stay informed.

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension and its standalone scripts are released under [Apache 2.0](LICENSE-mimic).  License files for dependencies and assets are located in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab builds upon the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following in academic publications:

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