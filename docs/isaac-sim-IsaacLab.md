![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Unleash the power of NVIDIA Isaac Sim with Isaac Lab, a GPU-accelerated, open-source framework designed to revolutionize your robotics research workflows.** [(Back to Original Repo)](https://github.com/isaac-sim/IsaacLab)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab provides a comprehensive environment for robotics research, specifically for Reinforcement Learning (RL), Imitation Learning, and Motion Planning, all built on NVIDIA Isaac Sim. It combines fast and accurate physics with advanced sensor simulation, making it perfect for sim-to-real transfer.  Its GPU acceleration significantly speeds up complex simulations and iterative processes.

## Key Features for Robotics Research

*   **Extensive Robot Models:** Includes 16 commonly available robot models (manipulators, quadrupeds, humanoids, etc.).
*   **Ready-to-Train Environments:** Over 30 pre-built environments compatible with popular RL frameworks like RSL RL, SKRL, RL Games, and Stable Baselines.  Supports Multi-Agent Reinforcement Learning (MARL).
*   **Realistic Physics Engine:** Utilizing rigid bodies, articulated systems, and deformable objects.
*   **Advanced Sensor Simulation:**  Supports a wide range of sensors, including RGB/depth/segmentation cameras, IMUs, and contact sensors.

## Getting Started

### Prerequisites
*   NVIDIA Isaac Sim ([Installation Instructions](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start))

### Installation Steps:

1.  **Clone Isaac Sim:**
    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**
    ```bash
    cd IsaacSim
    ./build.sh  # On Linux
    ./build.bat # On Windows
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

6.  **(Optional) Set up a Python Virtual Environment (e.g., using Conda):**
    *   **Linux:**
        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```
    *   **Windows:**
        ```bash
        _isaac_sim\setup_python_env.bat
        ```

7.  **Train Your First Model!**
    *   **Linux:**
        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```
    *   **Windows:**
        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation and Tutorials

Access comprehensive resources to quickly learn and utilize Isaac Lab:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Guide](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure compatibility by using the correct Isaac Sim version with your Isaac Lab release.

| Isaac Lab Version  | Isaac Sim Version |
| ------------------ | ----------------- |
| `main` branch      | Isaac Sim 4.5 / 5.0 |
| `v2.2.0`           | Isaac Sim 4.5 / 5.0 |
| `v2.1.1`           | Isaac Sim 4.5       |
| `v2.1.0`           | Isaac Sim 4.5       |
| `v2.0.2`           | Isaac Sim 4.5       |
| `v2.0.1`           | Isaac Sim 4.5       |
| `v2.0.0`           | Isaac Sim 4.5       |

## Contribute and Collaborate

Join the community and help improve Isaac Lab! Your contributions are welcome through:

*   **Bug Reports:** Help identify and fix issues.
*   **Feature Requests:** Suggest new functionalities.
*   **Code Contributions:** Enhance the framework directly.

Check out the [Contribution Guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for more information.

## Showcase Your Projects

Share your robotics projects, tutorials, and learning materials in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions to inspire others and foster collaboration within the community.

## Troubleshooting

Find solutions to common issues in the [Troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section. For problems related to Isaac Sim, consult its [Documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support and Community

*   **Discussions:** Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) to discuss ideas, ask questions, and request new features.
*   **Issues:** Report bugs, documentation problems, and feature requests using Github [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with NVIDIA Omniverse

To spotlight your work and explore collaboration opportunities, contact the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com. Engage with the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect with other developers and contribute to the ecosystem.

## License and Acknowledgements

Isaac Lab is licensed under the [BSD-3 License](LICENSE), while the `isaaclab_mimic` extension and its related scripts are under the [Apache 2.0](LICENSE-mimic) license.  The license files for dependencies are in the [`docs/licenses`](docs/licenses) directory.

Please cite the following paper in academic publications:

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