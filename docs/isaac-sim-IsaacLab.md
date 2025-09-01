![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab is a powerful, open-source framework built on NVIDIA Isaac Sim, designed to streamline robotics research workflows like reinforcement learning and sim-to-real transfer.**  Learn more and contribute on the original repository: [https://github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab offers a comprehensive suite of tools for advanced robotics simulation and development:

*   **Diverse Robot Models:**  Includes 16 pre-built robot models (manipulators, quadrupeds, humanoids) for various research applications.
*   **Extensive Environments:** Over 30 ready-to-train environments, compatible with popular RL frameworks like RSL RL, SKRL, RL Games, and Stable Baselines, including multi-agent RL support.
*   **Advanced Physics Simulation:** Supports rigid bodies, articulated systems, and deformable objects for realistic environment modeling.
*   **Realistic Sensor Simulation:** Provides RTX-based cameras (RGB, depth, segmentation), LIDAR, IMU, contact sensors, and ray casters for accurate sensor data generation.
*   **GPU-Accelerated Performance:** Enables faster simulation and computation, crucial for iterative processes such as reinforcement learning and data-intensive tasks.
*   **Flexible Deployment:** Supports local and cloud-based execution for scalability and versatility.

## Getting Started

Get up and running with Isaac Lab, starting with the installation of the open-source Isaac Sim:

### 1. Install Isaac Sim (if not already installed)

Follow the instructions in the [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) to install the base simulation environment.

### 2. Clone and Set up Isaac Lab

1.  Clone Isaac Lab

    ```bash
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

2.  Set up symlink:

    *   **Linux:**

        ```bash
        ln -s ../IsaacSim/_build/linux-x86_64/release _isaac_sim
        ```

    *   **Windows:**

        ```bash
        mklink /D _isaac_sim ..\IsaacSim\_build\windows-x86_64\release
        ```

3.  Install Isaac Lab

    *   **Linux:**

        ```bash
        ./isaaclab.sh -i
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -i
        ```

4.  **[Optional]** Set up a virtual python environment (e.g. for Conda):

    *   **Linux:**

        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```

    *   **Windows:**

        ```bash
        _isaac_sim\setup_python_env.bat
        ```

5.  Train!

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation

Comprehensive documentation is available to guide you through the use of Isaac Lab:

*   [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure compatibility by using the correct Isaac Sim version with your Isaac Lab release:

| Isaac Lab Version             | Isaac Sim Version   |
| ----------------------------- | ------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0 |
| `v2.1.X`                      | Isaac Sim 4.5       |
| `v2.0.X`                      | Isaac Sim 4.5       |

## Contribute

We encourage community contributions!  Please review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details on submitting bug reports, feature requests, and code contributions.

## Show & Tell: Share Your Projects!

Showcase your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell)  section of the `Discussions` area. Inspire others and foster innovation!

## Troubleshooting

Consult the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common solutions. For Isaac Sim-specific issues, refer to its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   **Discussions:** Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions, questions, and feature requests.
*   **Issues:**  Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) to report bugs and track specific work items.

## Connect with the NVIDIA Omniverse Community

Share your work and collaborate with the community!  Contact the NVIDIA Omniverse Community team at [OmniverseCommunity@nvidia.com](mailto:OmniverseCommunity@nvidia.com) or join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic).  Dependency and asset licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab builds upon the [Orbit](https://isaac-orbit.github.io/) framework.  Please cite it in academic publications:

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