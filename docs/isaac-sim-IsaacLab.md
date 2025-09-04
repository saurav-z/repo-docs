<!-- Isaac Lab Banner -->
![Isaac Lab](docs/source/_static/isaaclab.jpg)

<!-- Badges (Consider moving these to the top, below the title) -->
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Isaac Lab is a powerful, open-source framework built on NVIDIA Isaac Sim, designed to streamline robotics research workflows, enabling faster, more accurate simulations for reinforcement learning, imitation learning, and motion planning.**  Explore the  [Isaac Lab repository](https://github.com/isaac-sim/IsaacLab) to unlock the potential of advanced robotics simulation.

## Key Features

*   **Diverse Robot Models:** Includes 16 pre-built robot models, covering manipulators, quadrupeds, and humanoids.
*   **Extensive Environments:** Offers over 30 ready-to-train environments compatible with popular reinforcement learning frameworks like RSL RL, SKRL, RL Games, and Stable Baselines, with support for multi-agent reinforcement learning.
*   **Realistic Physics:** Supports rigid bodies, articulated systems, and deformable objects for accurate simulations.
*   **Advanced Sensor Simulation:** Provides a wide array of sensor options including RGB/depth/segmentation cameras, IMUs, contact sensors, and ray casters for comprehensive data collection.
*   **GPU-Accelerated:** Leveraging NVIDIA Isaac Sim, simulations run faster, enabling faster iteration.
*   **Flexible Deployment:** Run simulations locally or in the cloud.

## Getting Started

### Prerequisites

Before you begin, ensure you have:

*   **NVIDIA Isaac Sim:** (Installation instructions below).
*   **Python 3.11+:**  (or as compatible with your Isaac Sim version).
*   **Linux or Windows:** Operating system.

### Installation

1.  **Install NVIDIA Isaac Sim:** (Open Source)

    Follow the instructions in the [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) to install and build Isaac Sim.

2.  **Clone Isaac Lab:**

    ```bash
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

5.  **(Optional) Set up a Virtual Python Environment (e.g., for Conda):**

    *   **Linux:**

        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```

    *   **Windows:**

        ```bash
        _isaac_sim\setup_python_env.bat
        ```

6.  **Train Your First Model!**

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation & Resources

*   **Comprehensive Documentation:** Get detailed instructions and guides.
    *   [Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
    *   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
    *   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
    *   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure you use the correct Isaac Sim version with your Isaac Lab release:

| Isaac Lab Version             | Isaac Sim Version   |
| ----------------------------- | ------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0 |
| `v2.1.X`                      | Isaac Sim 4.5       |
| `v2.0.X`                      | Isaac Sim 4.5       |

## Contribute

We welcome contributions!  Review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn how to contribute to Isaac Lab.  Your contributions can include bug reports, feature requests, or code contributions.

## Show & Tell: Share Your Work

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions. Inspire others and foster collaboration!

## Troubleshooting

*   Check the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common fixes.
*   Report issues via [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues).
*   For Isaac Sim-specific issues, refer to the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   **Discussions:** Use [GitHub Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for questions and feature requests.
*   **Issues:** Use [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports and actionable work items.

## Connect with the NVIDIA Omniverse Community

Share your projects with the NVIDIA Omniverse Community by reaching out to  OmniverseCommunity@nvidia.com. Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect with other developers.

## License

*   Isaac Lab: [BSD-3 License](LICENSE)
*   `isaaclab_mimic` extension and scripts: [Apache 2.0](LICENSE-mimic)
*   Dependencies: License files are in the [`docs/licenses`](docs/licenses) directory.

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