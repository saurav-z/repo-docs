![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim that simplifies robotics research workflows, enabling faster and more accurate simulations for reinforcement learning, imitation learning, and motion planning.  [Explore the Isaac Lab repository](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


## Key Features for Robotics Simulation

Isaac Lab offers a comprehensive suite of features for robotics researchers:

*   **Diverse Robot Models:** Includes 16 pre-built robot models, including manipulators, quadrupeds, and humanoids.
*   **Extensive Environments:** Provides over 30 ready-to-train environments compatible with popular reinforcement learning frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and multi-agent reinforcement learning.
*   **Realistic Physics:** Supports rigid bodies, articulated systems, and deformable objects for accurate simulation.
*   **Advanced Sensor Simulation:**  Simulates realistic sensor data with RTX-based cameras (RGB/depth/segmentation), LIDAR, IMUs, and contact sensors.
*   **GPU Acceleration:** Leverages GPU acceleration for faster simulation and computation, accelerating iterative processes like reinforcement learning and data-intensive tasks.
*   **Scalability:** Can be run locally or distributed across the cloud for large-scale deployments.

## Getting Started with Isaac Lab

Jumpstart your robotics research with these resources:

*   **Documentation:** Comprehensive [documentation](https://isaac-sim.github.io/IsaacLab) with tutorials and guides.
*   **Installation:** Step-by-step [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation).
*   **Reinforcement Learning:** Learn about [reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html) with Isaac Lab.
*   **Tutorials:**  Explore hands-on [tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html).
*   **Environments:** Explore the available [environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab is designed to work with specific versions of NVIDIA Isaac Sim. Here's a table mapping Isaac Lab releases to compatible Isaac Sim versions:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

*Note:* The `feature/isaacsim_5_0` branch is under active development and may have breaking changes. It requires the [Isaac Sim 5.0 branch](https://github.com/isaac-sim/IsaacSim) and instructions can be found within the `feature/isaacsim_5_0` branch's README.

## Contribute to Isaac Lab

We welcome community contributions!  Find out more about contributing by checking out the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Showcase Your Work

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions area.

## Troubleshooting and Support

*   Refer to the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section.
*   Report issues or request features on [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues).
*   Discuss ideas and ask questions on [GitHub Discussions](https://github.com/isaac-sim/IsaacLab/discussions).
*   For Isaac Sim-specific issues, consult the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com or join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License and Acknowledgements

Isaac Lab is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic). See the `docs/licenses` directory for licenses of dependencies.

If you use Isaac Lab in your academic work, please cite the following:

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