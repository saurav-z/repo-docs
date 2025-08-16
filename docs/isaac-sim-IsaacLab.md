[![Isaac Lab](docs/source/_static/isaaclab.jpg)](https://github.com/isaac-sim/IsaacLab)

# Isaac Lab: Accelerate Your Robotics Research with NVIDIA Isaac Sim

Isaac Lab is a cutting-edge, open-source framework built on NVIDIA Isaac Sim, empowering researchers to accelerate robotics workflows for reinforcement learning, imitation learning, and motion planning.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

*   **Rich Robot Models:** Explore and experiment with a diverse library of 16 pre-built robot models, including manipulators, quadrupeds, and humanoids.
*   **Pre-built Environments:** Train your models in over 30 ready-to-use environments compatible with popular RL frameworks like RSL RL, SKRL, RL Games, and Stable Baselines, including multi-agent reinforcement learning support.
*   **Advanced Physics Simulation:** Leverage accurate and fast physics simulation, supporting rigid bodies, articulated systems, and deformable objects.
*   **Realistic Sensor Simulation:** Utilize RTX-based cameras, LIDAR, IMU, contact sensors, and ray casters for accurate and detailed sensor data.
*   **GPU Acceleration:** Benefit from GPU acceleration to run complex simulations and computations faster, crucial for iterative processes like reinforcement learning.
*   **Flexible Deployment:** Run Isaac Lab locally or distribute it across the cloud for large-scale robotics research and development.

## Getting Started

### Prerequisites

*   **NVIDIA Isaac Sim:** Isaac Lab is built on top of NVIDIA Isaac Sim. Refer to the [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) for installation instructions, and be sure to use a compatible version (see "Isaac Sim Version Dependency" below).
*   **Python:** Python 3.11 is recommended.

### Installation

1.  **Clone Isaac Sim:**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh  # For Linux
    # Or, for Windows:
    # build.bat
    ```

3.  **Clone Isaac Lab:**

    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

4.  **Set up Symlink (Important for Isaac Sim Integration):**

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

6.  **(Optional) Set up a Virtual Python Environment (e.g., Conda):**

    *   **Linux:**

        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```

    *   **Windows:**

        ```bash
        _isaac_sim\setup_python_env.bat
        ```

7.  **Run a Training Example:**

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

## Documentation

Access comprehensive documentation, including tutorials and guides, to help you get started:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Overview](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

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

Join our community and contribute to the development of Isaac Lab! We welcome contributions in the form of bug reports, feature requests, and code contributions. See our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details.

## Show & Tell: Share Your Work

Showcase your projects and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the `Discussions` area. Share tutorials, learning content, and exciting projects.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section, or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues). For issues related to Isaac Sim, refer to its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussions, questions, and feature requests.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) to track executable work with a defined scope.

## Connect with the NVIDIA Omniverse Community

Share your projects with the NVIDIA Omniverse Community! Contact OmniverseCommunity@nvidia.com to spotlight your work. Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) for collaboration and growth.

## License

Isaac Lab is released under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension and related scripts are under the [Apache 2.0](LICENSE-mimic) license. License files for dependencies are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is built upon the [Orbit](https://isaac-orbit.github.io/) framework.  Please cite it in academic publications:

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