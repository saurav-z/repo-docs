![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with NVIDIA Isaac Sim

**Isaac Lab is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim, designed to streamline robotics research workflows like reinforcement learning and sim-to-real transfer.**  Learn more about Isaac Lab on its [original repository](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab leverages the power of NVIDIA Isaac Sim to provide a robust platform for robotics research, offering fast and accurate physics simulation. It includes essential features for sensor simulation, such as RTX-based cameras, LIDAR, and contact sensors, enabling efficient sim-to-real transfer. Its GPU acceleration significantly speeds up complex simulations and iterative processes like reinforcement learning.  It can be run locally or on the cloud, offering flexibility for various deployments.

## Key Features

*   **Robots:** A diverse collection of 16 pre-configured robot models, including manipulators, quadrupeds, and humanoids.
*   **Environments:** Ready-to-train implementations of over 30 environments compatible with popular reinforcement learning frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and support for multi-agent reinforcement learning.
*   **Physics:** Accurate simulation of rigid bodies, articulated systems, and deformable objects.
*   **Sensors:** Comprehensive sensor suite, including RGB/depth/segmentation cameras, IMU, contact sensors, and ray casters.

## Getting Started

### Prerequisites

*   **NVIDIA Isaac Sim:** Requires a compatible version of NVIDIA Isaac Sim.  See the [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) for detailed installation instructions.

### Installation Steps

1.  **Clone Isaac Sim:**
    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```
2.  **Build Isaac Sim:**
    ```bash
    cd IsaacSim
    ./build.sh  # Or build.bat on Windows
    ```
3.  **Clone Isaac Lab:**
    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```
4.  **Set up Symlink:**
    *   **Linux:**
        ```bash
        ln -s ../IsaacSim/_build/linux-x86_64/release _isaac_sim
        ```
    *   **Windows:**
        ```bash
        mklink /D _isaac_sim ..\IsaacSim\_build\windows-x86_64\release
        ```
5.  **Install Isaac Lab:**
    *   **Linux:**
        ```bash
        ./isaaclab.sh -i
        ```
    *   **Windows:**
        ```bash
        isaaclab.bat -i
        ```
6.  **Optional: Set up a Python Virtual Environment (e.g., Conda):**
    *   **Linux:**
        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```
    *   **Windows:**
        ```bash
        _isaac_sim\setup_python_env.bat
        ```
7.  **Train!**
    *   **Linux:**
        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```
    *   **Windows:**
        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation

Comprehensive documentation is available to guide you through installation, reinforcement learning, tutorials, and available environments.

*   [Installation Steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

Ensure compatibility by using the correct Isaac Sim version for your Isaac Lab release.

| Isaac Lab Version             | Isaac Sim Version   |
| ----------------------------- | ------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0 |
| `v2.1.X`                      | Isaac Sim 4.5       |
| `v2.0.X`                      | Isaac Sim 4.5       |

## Contributing

We welcome contributions from the community!  See our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details on how to contribute.

## Show & Tell: Share Your Projects

Share your tutorials, learning content, and projects in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions.  Inspire others and contribute to the community!

## Troubleshooting

Refer to the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common fixes.  For Isaac Sim-specific issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).  Submit issues on [GitHub](https://github.com/isaac-sim/IsaacLab/issues).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussions, questions, and feature requests.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports, documentation issues, new features, or general updates.

## Connect with the NVIDIA Omniverse Community

Contact the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to spotlight your work.  Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect with developers and contribute.

## License

Isaac Lab is released under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension and its associated scripts are released under [Apache 2.0](LICENSE-mimic).  License files for dependencies and assets are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development originated from the [Orbit](https://isaac-orbit.github.io/) framework.  Please cite it in academic publications:

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