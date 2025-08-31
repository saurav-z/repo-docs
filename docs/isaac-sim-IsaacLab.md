![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with NVIDIA Isaac Sim

**Isaac Lab is a powerful, open-source framework built on NVIDIA Isaac Sim, streamlining robotics research workflows for reinforcement learning, imitation learning, and more.**  Explore the power of simulation and unlock the potential of your robotics projects!  [Go to the original repository](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab, built on NVIDIA Isaac Sim, offers a comprehensive platform for robotics researchers, providing:

*   **GPU-Accelerated Performance:**  Run complex simulations and computations faster with GPU acceleration, crucial for iterative processes like reinforcement learning.
*   **Sim-to-Real Transfer:** Leverage fast and accurate physics and sensor simulation for effective sim-to-real transfer in robotics.
*   **Versatile Deployment:** Run locally or in the cloud, offering flexibility for various project scales.

## Key Features

Isaac Lab equips you with a wide array of tools for advanced robotics research:

*   **Extensive Robot Models:** Includes 16 commonly available robot models, covering manipulators, quadrupeds, and humanoids.
*   **Diverse Simulation Environments:** Offers over 30 ready-to-train environments, compatible with leading reinforcement learning frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and supports multi-agent reinforcement learning.
*   **Advanced Physics Engine:** Provides support for rigid bodies, articulated systems, and deformable objects.
*   **Realistic Sensor Simulation:** Includes support for a range of sensors, such as RGB/depth/segmentation cameras, IMU, contact sensors, and ray casters.

## Getting Started

Get up and running with Isaac Lab quickly:

### Prerequisites

*   **NVIDIA Isaac Sim:** [Refer to the Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) for detailed installation instructions.

### Installation Steps

1.  **Clone Isaac Sim:**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh  # On Linux
    # OR
    build.bat   # On Windows
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

6.  **(Optional) Set up a Virtual Python Environment:**  (e.g., for Conda)

    *   **Linux:**

        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```

    *   **Windows:**

        ```bash
        _isaac_sim\setup_python_env.bat
        ```

7.  **Train Your Models!**

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation and Resources

*   **Comprehensive Documentation:** Explore detailed tutorials, step-by-step guides, and more on the [documentation page](https://isaac-sim.github.io/IsaacLab).
*   **Installation:** [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   **Reinforcement Learning:** [RL Examples](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   **Tutorials:** [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   **Available Environments:** [Environment Overview](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure compatibility with specific Isaac Sim versions for optimal performance.

| Isaac Lab Version | Isaac Sim Version |
| :---------------- | :---------------- |
| `main`            | 4.5 / 5.0         |
| `v2.2.X`          | 4.5 / 5.0         |
| `v2.1.X`          | 4.5               |
| `v2.0.X`          | 4.5               |

## Contributing

We welcome community contributions! Check our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details on bug reports, feature requests, and code submissions.

## Share Your Work: Show & Tell

Showcase your projects and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section within the `Discussions` area.  Share tutorials, learning content, and project demos to contribute to the robotics community.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues). For Isaac Sim-specific issues, refer to its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   **Discussions:**  Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussions, questions, and feature requests.
*   **Issues:** Report bugs, documentation issues, new features, and updates through GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with the NVIDIA Omniverse Community

Share your projects and collaborate with the community!  Contact the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com or join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and associated scripts are under the [Apache 2.0](LICENSE-mimic) license.  License files for dependencies are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab builds upon the foundation of the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in academic publications:

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