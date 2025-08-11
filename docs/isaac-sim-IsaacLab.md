![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is a powerful, open-source framework for robotics research, built upon NVIDIA Isaac Sim, designed to simplify your workflows in reinforcement learning, imitation learning, and motion planning.  Explore the full potential of robotics simulation with **[Isaac Lab](https://github.com/isaac-sim/IsaacLab)**.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

*   **GPU-Accelerated Simulation:** Leverage the power of NVIDIA GPUs for faster, more efficient simulations.
*   **Extensive Robot Models:** Access a diverse collection of 16+ pre-built robot models, including manipulators, quadrupeds, and humanoids.
*   **Ready-to-Train Environments:** Utilize over 30 pre-configured environments compatible with popular RL frameworks like RSL RL, SKRL, RL Games, and Stable Baselines, with support for multi-agent RL.
*   **Advanced Physics and Sensors:** Benefit from accurate physics simulation, including rigid bodies, articulated systems, and deformable objects, along with realistic sensor simulation (RGB/depth cameras, LIDAR, IMU, etc.).
*   **Sim-to-Real Transfer:** Designed to facilitate seamless sim-to-real transfer for your robotics applications.

## Getting Started

### Prerequisites

*   **NVIDIA Isaac Sim:** Ensure you have NVIDIA Isaac Sim installed.  Refer to the [Isaac Sim Quick Start Guide](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) for installation instructions.
*   **Python:** Python 3.11 is recommended.

### Installation Steps

1.  **Clone Isaac Sim:**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh  # Linux
    # or
    .\build.bat  # Windows
    ```

3.  **Clone Isaac Lab:**

    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

4.  **Set up Symlink (Important):**

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

6.  **(Optional) Set up a Virtual Python Environment (Recommended):**

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

### Documentation

Explore our comprehensive documentation for detailed guides, tutorials, and environment overviews:

*   [Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab requires specific versions of Isaac Sim.  Here's a compatibility table:

| Isaac Lab Version     | Isaac Sim Version   |
| --------------------- | ------------------- |
| `main`                | 4.5 / 5.0         |
| `v2.2.0`              | 4.5 / 5.0         |
| `v2.1.1`              | 4.5               |
| `v2.1.0`              | 4.5               |
| `v2.0.2`              | 4.5               |
| `v2.0.1`              | 4.5               |
| `v2.0.0`              | 4.5               |

## Contributing

We warmly welcome contributions!  Review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn how you can help improve Isaac Lab.

## Showcase Your Work: Show & Tell

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section to inspire others and foster collaboration.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).  For Isaac Sim-specific problems, consult the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general questions and discussions.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports, documentation issues, and feature requests.

## Connect with the NVIDIA Omniverse Community

Share your projects and collaborate with the community by contacting the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com, and join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its related scripts are licensed under [Apache 2.0](LICENSE-mimic). Dependency licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab builds on the foundation of the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following in academic publications:

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