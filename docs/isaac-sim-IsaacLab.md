![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Revolutionizing Robotics Research with GPU-Accelerated Simulation

**Accelerate your robotics research with Isaac Lab, a powerful, open-source framework built on NVIDIA Isaac Sim for efficient and accurate simulation and sim-to-real transfer.**  ([See the original repository](https://github.com/isaac-sim/IsaacLab))

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab is an open-source framework meticulously designed to streamline robotics research workflows. It is built upon the robust foundation of [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html), and offers a powerful combination of fast, accurate physics and sensor simulation. This makes it an ideal choice for researchers working on reinforcement learning, imitation learning, motion planning, and sim-to-real transfer.

Key features of Isaac Lab include:

*   **Diverse Robot Models:** Explore 16 pre-built robot models, including manipulators, quadrupeds, and humanoids.
*   **Extensive Environments:** Train your models in over 30 ready-to-use environments, compatible with leading RL frameworks like RSL RL, SKRL, RL Games, and Stable Baselines. Multi-agent reinforcement learning is also supported.
*   **Realistic Physics:** Leverage rigid bodies, articulated systems, and deformable objects for highly accurate simulations.
*   **Advanced Sensor Simulation:** Utilize RTX-based cameras, LIDAR, and contact sensors to gather data as in the real world.
*   **GPU Acceleration:** Run complex simulations and computations significantly faster for iterative processes, such as reinforcement learning.
*   **Cloud and Local Deployment:** Run Isaac Lab locally or on the cloud for flexible large-scale deployments.

## Getting Started

### Prerequisites
*   Python 3.11
*   NVIDIA Isaac Sim (See below)

### Installation

1.  **Install NVIDIA Isaac Sim:**
    *   Refer to the official [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) for detailed installation instructions.

2.  **Clone Isaac Sim:**
    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

3.  **Build Isaac Sim:**
    ```bash
    cd IsaacSim
    ./build.sh  # On Linux
    ./build.bat # On Windows
    ```

4.  **Clone Isaac Lab:**
    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

5.  **Set up symlink:**
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

7.  **(Optional) Set up a virtual Python environment:**
    *   **Linux (using Conda):**
        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```
    *   **Windows:**
        ```bash
        _isaac_sim\setup_python_env.bat
        ```

8.  **Train your first model:**
    *   **Linux:**
        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```
    *   **Windows:**
        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

## Documentation

Comprehensive documentation is available to guide you through the framework:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Step-by-Step Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab is built on top of Isaac Sim and requires specific versions of Isaac Sim that are compatible with each release of Isaac Lab.
Below, we outline the recent Isaac Lab releases and GitHub branches and their corresponding dependency versions for Isaac Sim.

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

Join the community and help improve Isaac Lab! We welcome contributions in the form of:

*   Bug reports
*   Feature requests
*   Code contributions

Review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for more details.

## Show & Tell: Share Your Projects

Showcase your work and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of our Discussions.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or submit an issue on [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or ask on the [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions, questions, and feature requests.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for reporting bugs, documentation issues, or tracking specific tasks.

## Connect with the NVIDIA Omniverse Community

Share your projects and connect with the community. Contact the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com. Also, join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect with other developers.

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

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