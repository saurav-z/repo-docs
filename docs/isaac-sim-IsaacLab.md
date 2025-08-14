[![Isaac Lab](docs/source/_static/isaaclab.jpg)](https://github.com/isaac-sim/IsaacLab)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

Isaac Lab is an open-source, GPU-accelerated robotics simulation framework built on NVIDIA Isaac Sim, simplifying and unifying workflows for reinforcement learning, imitation learning, and motion planning.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

*   **Diverse Robot Models:** Includes 16 commonly available robot models, such as manipulators, quadrupeds, and humanoids.
*   **Pre-Built Environments:**  Offers over 30 ready-to-train environments compatible with popular RL frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and supports multi-agent reinforcement learning.
*   **Advanced Physics Simulation:** Supports rigid bodies, articulated systems, and deformable objects for realistic robot behavior.
*   **Realistic Sensor Simulation:** Provides RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, and ray casters.
*   **GPU Acceleration:** Enables faster simulation and computation for iterative processes like RL, and data-intensive tasks.
*   **Cloud and Local Deployment:** Provides flexibility for large-scale deployments.

## Getting Started

To get started with Isaac Lab, you need to install both [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) and Isaac Lab.

### 1. Prerequisites: Install Isaac Sim

See the [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) for detailed installation instructions, which includes cloning and building Isaac Sim.

### 2. Clone and Set Up Isaac Lab

1.  Clone Isaac Lab:

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

3.  Install Isaac Lab:

    *   **Linux:**

        ```bash
        ./isaaclab.sh -i
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -i
        ```

4.  [Optional] Set up a virtual python environment (e.g. for Conda)

    *   **Linux:**

        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```

    *   **Windows:**

        ```bash
        _isaac_sim\setup_python_env.bat
        ```

5.  Train a model (Example):

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation

Comprehensive documentation, including tutorials and guides, is available at: [https://isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab)

*   [Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

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

Contributions are welcome! Review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details on how to contribute.

## Show & Tell

Share your projects and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions.

## Troubleshooting

Consult the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).
For Isaac Sim-specific issues, refer to its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions, questions, and feature requests.
*   Use Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) to report bugs, documentation issues, and for implementing new features.

## Connect with the NVIDIA Omniverse Community

Share your work with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com, or connect with other developers on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension is licensed under [Apache 2.0](LICENSE-mimic). Dependency and asset licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

This project is built on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following in academic publications:

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