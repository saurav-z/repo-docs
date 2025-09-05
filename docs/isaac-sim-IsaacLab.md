![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Isaac Lab** is a powerful, open-source framework built on NVIDIA Isaac Sim, designed to streamline robotics research, offering GPU-accelerated simulation for reinforcement learning, imitation learning, and more; visit the [original repo](https://github.com/isaac-sim/IsaacLab) to learn more!

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab leverages NVIDIA Isaac Sim to provide a robust platform for robotics research.  It features fast and accurate physics and sensor simulation, making it ideal for sim-to-real transfer.  The framework's GPU acceleration enables users to run complex simulations and computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks.

## Key Features

*   **Diverse Robot Models:** Access a wide array of robots, including manipulators, quadrupeds, and humanoids, with 16 commonly available models.
*   **Ready-to-Train Environments:** Utilize pre-built environments (30+), compatible with popular reinforcement learning frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and multi-agent reinforcement learning.
*   **Advanced Physics Engine:** Benefit from a robust physics engine supporting rigid bodies, articulated systems, and deformable objects.
*   **Comprehensive Sensor Suite:** Leverage a suite of sensors, including RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, and ray casters.

## Getting Started

### Prerequisites

*   NVIDIA Isaac Sim (Installation instructions below)
*   Python 3.11+
*   Linux or Windows 64-bit operating system

### Installation Guide

1.  **Install Isaac Sim:**
    *   Follow the detailed instructions in the [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start).
    *   Clone the Isaac Sim repository:
        ```bash
        git clone https://github.com/isaac-sim/IsaacSim.git
        ```
    *   Build Isaac Sim:
        ```bash
        cd IsaacSim
        ./build.sh  # Or build.bat on Windows
        ```

2.  **Clone Isaac Lab:**
    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

3.  **Set up Symlink:**
    *   **Linux:**
        ```bash
        ln -s ../IsaacSim/_build/linux-x86_64/release _isaac_sim
        ```
    *   **Windows:**
        ```bash
        mklink /D _isaac_sim ..\IsaacSim\_build\windows-x86_64\release
        ```

4.  **Install Isaac Lab:**
    *   **Linux:**
        ```bash
        ./isaaclab.sh -i
        ```
    *   **Windows:**
        ```bash
        isaaclab.bat -i
        ```

5.  **(Optional) Set Up Virtual Environment:**
    *   **Linux (Conda example):**
        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```
    *   **Windows:**
        ```bash
        _isaac_sim\setup_python_env.bat
        ```

6.  **Train a Model:**
    *   **Linux:**
        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```
    *   **Windows:**
        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation

Explore the [documentation page](https://isaac-sim.github.io/IsaacLab) for comprehensive information, including:

*   [Installation Steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Guide](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

Ensure compatibility by matching Isaac Lab versions with the correct Isaac Sim version:

| Isaac Lab Version             | Isaac Sim Version   |
| ----------------------------- | ------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0 |
| `v2.1.X`                      | Isaac Sim 4.5       |
| `v2.0.X`                      | Isaac Sim 4.5       |

## Contributing

We welcome community contributions!  Review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to get started.

## Show & Tell: Share Your Projects

Showcase your work and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions page. Share your tutorials, projects, and learning content.

## Troubleshooting

Refer to the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common solutions or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For Isaac Sim-related issues, consult the [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussions, questions, and feature requests.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports, documentation issues, and clear deliverables.

## Connect with the NVIDIA Omniverse Community

Share your project or resource with the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com, or join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension and its scripts are licensed under [Apache 2.0](LICENSE-mimic).  See the [`docs/licenses`](docs/licenses) directory for dependency licenses.

## Acknowledgement

Isaac Lab is built upon the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following in academic publications:

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