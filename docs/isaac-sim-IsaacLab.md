![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with Realistic Simulation

**Isaac Lab** is a powerful, open-source framework built on NVIDIA Isaac Sim, designed to streamline robotics research workflows by providing fast, accurate, and GPU-accelerated simulation capabilities.  [Visit the original repository](https://github.com/isaac-sim/IsaacLab)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab empowers robotics researchers to accelerate their work in reinforcement learning, imitation learning, and motion planning with its realistic physics and sensor simulation.  It leverages the power of NVIDIA Isaac Sim for accurate sensor data and utilizes GPU acceleration to speed up simulations and computations, making it ideal for iterative processes like reinforcement learning and data-intensive tasks. The framework supports both local and cloud-based deployments for maximum flexibility.

## Key Features

*   **Diverse Robot Models:** Access a library of 16 commonly available robot models, including manipulators, quadrupeds, and humanoids.
*   **Rich Environments:** Train your robots in over 30 ready-to-use environments compatible with popular reinforcement learning frameworks like RSL RL, SKRL, RL Games, and Stable Baselines. Multi-agent reinforcement learning is also supported.
*   **Realistic Physics:**  Simulate rigid bodies, articulated systems, and deformable objects for accurate robot interactions.
*   **Advanced Sensors:**  Utilize a wide range of sensors, including RGB/depth/segmentation cameras, IMUs, contact sensors, and ray casters to provide comprehensive environmental data.

## Getting Started

Get up and running with Isaac Lab quickly using our comprehensive documentation.

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Resources](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Step-by-Step Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab requires specific versions of Isaac Sim.  Below are the compatible versions for each Isaac Lab release:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.1`                      | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

The `feature/isaacsim_5_0` branch is under active development and may have breaking changes. It requires the [Isaac Sim 5.0 branch](https://github.com/isaac-sim/IsaacSim). Refer to the `feature/isaacsim_5_0` branch README for instructions.

## Contribute to Isaac Lab

Your contributions are welcome!  Help us improve Isaac Lab by submitting bug reports, feature requests, or code contributions.  See our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Share Your Projects

Showcase your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of our Discussions. Inspire others and contribute to the community!

## Troubleshooting

Find solutions to common issues in our [troubleshooting guide](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues). For Isaac Sim-specific issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions, questions, and feature requests.
*   Use Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) to track specific work with a clear scope and deliverable (bug fixes, documentation, new features, updates).

## Connect with the NVIDIA Omniverse Community

Share your projects with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com, and join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is under [Apache 2.0](LICENSE-mimic).  License files for dependencies and assets are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is developed from the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in academic publications:

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