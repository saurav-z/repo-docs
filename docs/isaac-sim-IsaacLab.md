![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Isaac Lab** is a powerful, open-source framework built on NVIDIA Isaac Sim, designed to streamline robotics research through high-fidelity simulation and rapid iteration.  [Visit the original repository on GitHub](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab empowers robotics researchers with a comprehensive suite of tools:

*   **Extensive Robot Library:** Access a diverse range of 16 pre-configured robot models, from manipulators to humanoids, ready for simulation.
*   **Rich Environment Support:**  Train your robots in over 30 pre-built environments, seamlessly integrating with popular RL frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and supporting multi-agent learning.
*   **Advanced Physics Simulation:** Leverage accurate physics models for rigid bodies, articulated systems, and deformable objects.
*   **Realistic Sensor Simulation:** Utilize RTX-based cameras, LIDAR, IMU, contact sensors, and ray casters to create realistic sensor data for your simulated robots.
*   **GPU-Accelerated Performance:** Experience faster simulations and computations thanks to GPU acceleration, crucial for iterative research processes.
*   **Flexible Deployment:** Run your simulations locally or in the cloud, offering scalability for any project.

## Getting Started

### Prerequisites

*   **NVIDIA Isaac Sim:** Isaac Lab is built on Isaac Sim.  Install Isaac Sim first. You can find detailed instructions in the [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start).

### Installation

1.  **Clone Isaac Sim:**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh  #  For Linux
    #  OR
    ./build.bat # For Windows
    ```

3.  **Clone Isaac Lab:**

    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

4.  **Set up symlink in Isaac Lab:**

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

6.  **(Optional) Set up a virtual Python environment:**

    *   **Linux (Conda Example):**

        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```

    *   **Windows:**

        ```bash
        _isaac_sim\setup_python_env.bat
        ```

7.  **Train your first robot!**

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation

For comprehensive guides and tutorials, explore the official documentation:

*   [Installation Steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Guides](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure you use the correct Isaac Sim version with your Isaac Lab release:

| Isaac Lab Version             | Isaac Sim Version   |
| ----------------------------- | ------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 |
| `v2.2.0`                      | Isaac Sim 4.5 / 5.0 |
| `v2.1.1`                      | Isaac Sim 4.5       |
| `v2.1.0`                      | Isaac Sim 4.5       |
| `v2.0.2`                      | Isaac Sim 4.5       |
| `v2.0.1`                      | Isaac Sim 4.5       |
| `v2.0.0`                      | Isaac Sim 4.5       |

## Contribute

We welcome community contributions!  Review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details on how to contribute code, report bugs, and suggest features.

## Show & Tell: Share Your Projects

Showcase your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of our Discussions area. Share your work to inspire others and contribute to a collaborative learning environment.

## Troubleshooting

*   Consult the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common solutions.
*   For Isaac Sim-related issues, refer to the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or the [Isaac Sim forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).
*   For general issues, use the GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues).
*   For discussions, ask questions, and new features, use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions, questions, and feature requests.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) to report bugs or for executable project issues.

## Connect with the NVIDIA Omniverse Community

Share your projects and resources by contacting the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com.

Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect with other developers and build a collaborative ecosystem.

## License

Isaac Lab is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its associated scripts are released under the [Apache 2.0](LICENSE-mimic) license. License files for dependencies and assets are found in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is built upon the foundation of the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following paper in academic publications:

```bibtex
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