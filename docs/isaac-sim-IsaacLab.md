<!-- Improved & Summarized README for Isaac Lab -->
![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Isaac Lab provides a powerful, open-source framework built on NVIDIA Isaac Sim to streamline robotics research, enabling rapid prototyping and sim-to-real transfer.**  Explore the original repository on [GitHub](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab simplifies robotics research with these core capabilities:

*   **Robotics**: Includes a diverse range of 16 pre-built robot models (manipulators, quadrupeds, humanoids).
*   **Environments**: Offers over 30 ready-to-train environments, compatible with popular RL frameworks (RSL RL, SKRL, RL Games, Stable Baselines).  Supports multi-agent reinforcement learning.
*   **Physics**: Incorporates realistic physics simulation for rigid bodies, articulated systems, and deformable objects.
*   **Sensors**: Provides a suite of advanced sensors, including RTX-based cameras, LiDAR, IMU, and contact sensors for accurate data acquisition.

## Getting Started

Follow these steps to begin using Isaac Lab:

1.  **Install Isaac Sim:** Refer to the [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) for detailed instructions.

2.  **Clone Isaac Sim:**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

3.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh  # or build.bat for Windows
    ```

4.  **Clone Isaac Lab:**

    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

5.  **Set up Symlink:**

    *   **Linux:**

        ```bash
        ln -s ../IsaacSim/_build/linux-x86_64/release _isaac_sim
        ```
    *   **Windows:**

        ```bash
        mklink /D _isaac_sim ..\IsaacSim\_build\windows-x86_64\release
        ```

6.  **Install Isaac Lab:**

    *   **Linux:**

        ```bash
        ./isaaclab.sh -i
        ```
    *   **Windows:**

        ```bash
        isaaclab.bat -i
        ```

7.  **(Optional) Set up a Virtual Environment (e.g., Conda):**  Follow instructions in `_isaac_sim/setup_conda_env.sh` (Linux) or `_isaac_sim\setup_python_env.bat` (Windows).

8.  **Train a Model:**

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```
    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

## Documentation & Resources

*   **Comprehensive Documentation:** Access detailed tutorials and guides on our [documentation page](https://isaac-sim.github.io/IsaacLab).
*   **Key Documentation Sections:**
    *   [Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
    *   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
    *   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
    *   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure compatibility by using the correct Isaac Sim version with your Isaac Lab release.  Refer to this table:

| Isaac Lab Version       | Isaac Sim Version   |
| ----------------------- | ------------------- |
| `main` branch           | Isaac Sim 4.5 / 5.0 |
| `v2.2.0`                | Isaac Sim 4.5 / 5.0 |
| `v2.1.1`                | Isaac Sim 4.5       |
| `v2.1.0`                | Isaac Sim 4.5       |
| `v2.0.2`                | Isaac Sim 4.5       |
| `v2.0.1`                | Isaac Sim 4.5       |
| `v2.0.0`                | Isaac Sim 4.5       |

## Contributing

We welcome contributions from the community!  See our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details on how to contribute.

## Show & Tell: Share Your Projects

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section to inspire others.

## Troubleshooting

Consult the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common issues.  For Isaac Sim-specific problems, check its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general inquiries and feature requests.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for specific bug reports and actionable tasks.

## Connect with the NVIDIA Omniverse Community

Share your projects and connect with the community by contacting OmniverseCommunity@nvidia.com or joining the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic). License files for dependencies are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is built on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following paper in your academic work:

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