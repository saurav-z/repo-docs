![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Isaac Lab is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim, streamlining robotics research workflows for reinforcement learning, imitation learning, and motion planning; [Explore the original repository](https://github.com/isaac-sim/IsaacLab).**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab provides researchers and developers with a powerful platform for robotics simulation,
leveraging the capabilities of NVIDIA Isaac Sim for fast and accurate physics and sensor simulation.
It's designed to bridge the gap between simulation and real-world robotics applications (sim-to-real).

## Key Features

*   **Rich Robot Models:** Includes a diverse library of 16 commonly available robot models, from manipulators to humanoids.
*   **Pre-built Environments:** Offers over 30 ready-to-train environments compatible with popular reinforcement learning frameworks (RSL RL, SKRL, RL Games, Stable Baselines), including multi-agent scenarios.
*   **Advanced Physics Engine:** Supports rigid bodies, articulated systems, and deformable objects for realistic simulations.
*   **Comprehensive Sensor Suite:** Provides a range of sensors including RGB/depth/segmentation cameras, IMUs, contact sensors, and ray casters.

## Getting Started

### Prerequisites

*   **NVIDIA Isaac Sim:** Requires a compatible version of NVIDIA Isaac Sim (see version compatibility table below).
*   **Python 3.11+:**  Ensure Python 3.11 or a later version is installed.
*   **Operating System:**  Linux or Windows 64-bit.

### Installation

1.  **Clone Isaac Sim:**
    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```
2.  **Build Isaac Sim:**
    ```bash
    cd IsaacSim
    ./build.sh  # For Linux
    # or
    build.bat  # For Windows
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
6.  **(Optional) Set up a Virtual Python Environment:**
    *   **Linux (Conda example):**
        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```
    *   **Windows:**
        ```bash
        _isaac_sim\setup_python_env.bat
        ```
7.  **Train Your First Agent:**
    *   **Linux:**
        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```
    *   **Windows:**
        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation

*   **[Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)**
*   **[Reinforcement Learning Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)**
*   **[Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)**
*   **[Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)**

## Isaac Sim Version Dependency

Maintain version compatibility with the following table:

| Isaac Lab Version    | Isaac Sim Version |
| -------------------- | ----------------- |
| `main` branch        | Isaac Sim 4.5 / 5.0 |
| `v2.2.X`             | Isaac Sim 4.5 / 5.0 |
| `v2.1.X`             | Isaac Sim 4.5       |
| `v2.0.X`             | Isaac Sim 4.5       |

## Contributing

We welcome contributions!  Review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details.

## Show & Tell - Share Your Projects!

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions. Inspire others and contribute to the community!

## Troubleshooting

Refer to the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For Isaac Sim-specific issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   **Discussions:** Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for questions and feature requests.
*   **Issues:** Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports and specific tasks.

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com.

Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect, collaborate, and grow the community.

## License

Isaac Lab is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its scripts are under the [Apache 2.0](LICENSE-mimic) license.  License files for dependencies are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is built on the foundation of the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in your publications:

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