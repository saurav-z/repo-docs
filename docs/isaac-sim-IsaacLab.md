![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Isaac Lab is a cutting-edge, open-source framework that empowers robotics researchers with GPU-accelerated simulation capabilities, built upon NVIDIA Isaac Sim.**  [See the original repository](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab streamlines robotics research workflows by providing fast, accurate physics and sensor simulation, making it ideal for:

*   **Reinforcement Learning (RL)**
*   **Imitation Learning**
*   **Motion Planning**
*   **Sim-to-Real Transfer**

Leverage the power of NVIDIA Isaac Sim to build accurate simulations and bridge the gap between virtual and real-world robotics.

## Key Features of Isaac Lab

Isaac Lab provides a comprehensive set of tools and environments designed to facilitate robot learning:

*   **Diverse Robot Models:** Access a wide range of robots, including manipulators, quadrupeds, and humanoids (16+ models).
*   **Ready-to-Train Environments:** Utilize 30+ pre-built environments compatible with popular RL frameworks such as RSL RL, SKRL, RL Games, and Stable Baselines. Support for multi-agent reinforcement learning.
*   **Advanced Physics Simulation:** Simulate rigid bodies, articulated systems, and deformable objects with high fidelity.
*   **Realistic Sensor Suite:** Access a full suite of sensors, including RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, and ray casters.
*   **GPU-Accelerated Performance:** Accelerate complex simulations and computations for faster iteration in RL and data-intensive tasks.
*   **Flexible Deployment:** Run simulations locally or distribute them across the cloud for large-scale experiments.

## Getting Started

### Prerequisites

*   Ensure you have NVIDIA Isaac Sim installed.  See [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) for installation instructions.

### Installation Steps

1.  **Clone Isaac Sim:**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh
    ```

    *   On Windows, use `build.bat`.

3.  **Clone Isaac Lab:**

    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

4.  **Set up symlink:**

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

    *   **Linux (e.g., for Conda):**

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

*   Explore our comprehensive [documentation](https://isaac-sim.github.io/IsaacLab) to get started:
    *   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
    *   [Reinforcement Learning Overview](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
    *   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
    *   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

Maintain compatibility with the right version of Isaac Sim:

| Isaac Lab Version | Isaac Sim Version |
| ----------------- | ----------------- |
| `main`            | Isaac Sim 4.5 / 5.0 |
| `v2.2.X`          | Isaac Sim 4.5 / 5.0 |
| `v2.1.X`          | Isaac Sim 4.5       |
| `v2.0.X`          | Isaac Sim 4.5       |

## Contributing

We welcome community contributions!  Refer to our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details on how to contribute.

## Show & Tell: Share Your Projects

Showcase your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions.  Inspire others and contribute to the community!

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section.  If you encounter an issue, please [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, please check the [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussions, questions, and feature requests.
*   Use Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) for executable pieces of work with a definite scope.

## Connect with the NVIDIA Omniverse Community

Share your projects and resources by contacting the NVIDIA Omniverse Community team at [OmniverseCommunity@nvidia.com](mailto:OmniverseCommunity@nvidia.com).  Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect with other developers.

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic).  License files for dependencies are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab originates from the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in academic publications:

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