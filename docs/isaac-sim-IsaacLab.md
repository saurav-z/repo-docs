![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab is a powerful, open-source framework built on NVIDIA Isaac Sim, designed to streamline robotics research workflows like reinforcement learning, imitation learning, and motion planning.**  Check out the [original repository](https://github.com/isaac-sim/IsaacLab) for more details!

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab offers a comprehensive suite of tools and environments for robotics research, including:

*   **GPU Acceleration:** Leverage the power of your GPU for faster simulations and computations, crucial for iterative processes like reinforcement learning.
*   **Diverse Robot Models:** Explore a wide range of robots, including manipulators, quadrupeds, and humanoids, with 16 pre-built models.
*   **Rich Environments:** Utilize ready-to-train implementations of 30+ environments, compatible with popular reinforcement learning frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and supporting multi-agent reinforcement learning.
*   **Realistic Physics:** Simulate rigid bodies, articulated systems, and deformable objects for accurate and immersive environments.
*   **Advanced Sensors:** Utilize realistic RGB/depth/segmentation cameras, IMUs, contact sensors, and ray casters to gather valuable data.
*   **Flexible Deployment:** Run simulations locally or scale them across the cloud for large-scale projects.

## Getting Started

Follow these steps to get started with Isaac Lab:

1.  **Install Isaac Sim:** (See [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) for detailed instructions).
    *   Clone Isaac Sim: `git clone https://github.com/isaac-sim/IsaacSim.git`
    *   Build Isaac Sim: `cd IsaacSim && ./build.sh` (Linux) or `build.bat` (Windows)
2.  **Clone Isaac Lab:**
    *   `cd .. && git clone https://github.com/isaac-sim/IsaacLab.git && cd isaaclab`
3.  **Set Up Symlink:**  (Adapt the command to your operating system).
    *   Linux: `ln -s ../IsaacSim/_build/linux-x86_64/release _isaac_sim`
    *   Windows: `mklink /D _isaac_sim ..\IsaacSim\_build\windows-x86_64\release`
4.  **Install Isaac Lab:**  (Adapt the command to your operating system).
    *   Linux: `./isaaclab.sh -i`
    *   Windows: `isaaclab.bat -i`
5.  **(Optional) Set Up Virtual Environment:** (Adapt the command to your operating system)
    *   Linux: `source _isaac_sim/setup_conda_env.sh`
    *   Windows: `_isaac_sim\setup_python_env.bat`
6.  **Train Your First Model!** (Adapt the command to your operating system)
    *   Linux: `./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless`
    *   Windows: `isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless`

## Documentation

Explore the comprehensive documentation for detailed tutorials and guides:

*   [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure compatibility by using the correct Isaac Sim version for your Isaac Lab release:

| Isaac Lab Version             | Isaac Sim Version   |
| ----------------------------- | ------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 |
| `v2.2.0`                      | Isaac Sim 4.5 / 5.0 |
| `v2.1.1`                      | Isaac Sim 4.5       |
| `v2.1.0`                      | Isaac Sim 4.5       |
| `v2.0.2`                      | Isaac Sim 4.5       |
| `v2.0.1`                      | Isaac Sim 4.5       |
| `v2.0.0`                      | Isaac Sim 4.5       |

## Contributing

We welcome community contributions!  Refer to the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details on how to contribute.

## Show & Tell

Share your projects and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section. Report issues or ask questions on the [issues page](https://github.com/isaac-sim/IsaacLab/issues).

For Isaac Sim-related issues, consult the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   **Discussions:** Use [GitHub Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions, questions, and feature requests.
*   **Issues:**  Use [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues) for tracking specific bugs, documentation errors, and feature implementations.

## Connect with the NVIDIA Omniverse Community

Share your work with the Omniverse Community: [OmniverseCommunity@nvidia.com](mailto:OmniverseCommunity@nvidia.com). Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect with other developers.

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic).  See the [`docs/licenses`](docs/licenses) directory for dependency licenses.

## Acknowledgement

Isaac Lab is built on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following paper in your academic publications:

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