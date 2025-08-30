![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with NVIDIA Isaac Sim

**Isaac Lab** is an open-source framework built on NVIDIA Isaac Sim that simplifies robotics research workflows, providing GPU-accelerated simulation for reinforcement learning, imitation learning, and motion planning.  Explore the code on the [original repository](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


## Key Features

Isaac Lab offers a comprehensive suite of tools designed to supercharge your robotics research:

*   **Diverse Robot Models:**  Access a library of 16 pre-built robot models, including manipulators, quadrupeds, and humanoids.
*   **Ready-to-Train Environments:** Utilize over 30 pre-configured environments compatible with popular reinforcement learning frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and multi-agent reinforcement learning.
*   **Advanced Physics Simulation:** Leverage robust physics engines to simulate rigid bodies, articulated systems, and deformable objects.
*   **Realistic Sensor Simulation:** Implement accurate RGB/depth/segmentation cameras, camera annotations, IMUs, contact sensors, and ray casters.
*   **GPU Acceleration:** Benefit from GPU-accelerated simulations, enabling faster training and data-intensive tasks.
*   **Cloud-Ready Deployment:** Run simulations locally or distribute them across the cloud for large-scale experiments.

## Getting Started

### 1. Prerequisites

*   **NVIDIA Isaac Sim:** Isaac Lab is built upon NVIDIA Isaac Sim. Install the necessary version by following the instructions in the [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start).

### 2. Installation

1.  **Clone Isaac Sim:**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh  # For Linux
    # Use build.bat for Windows
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

6.  **(Optional) Set up a Python Virtual Environment (e.g., Conda):**

    *   **Linux:**

        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```

    *   **Windows:**

        ```bash
        _isaac_sim\setup_python_env.bat
        ```

### 3. Run a Training Example

1.  **Training Example:**

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### 4. Explore the Documentation

*   Access comprehensive tutorials and guides on the [documentation page](https://isaac-sim.github.io/IsaacLab):
    *   [Installation Steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
    *   [Reinforcement Learning Examples](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
    *   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
    *   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure compatibility by using the correct Isaac Sim version with your Isaac Lab release:

| Isaac Lab Version             | Isaac Sim Version   |
| ----------------------------- | ------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0 |
| `v2.1.X`                      | Isaac Sim 4.5       |
| `v2.0.X`                      | Isaac Sim 4.5       |

## Contributing

We welcome community contributions! Please review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details on how to contribute.  Contributions can include bug reports, feature requests, and code contributions.

## Show & Tell: Share Your Robotics Innovations

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions tab to inspire others and foster collaboration.

## Troubleshooting

For common issues and solutions, see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section. If you encounter issues, please [submit an issue](https://github.com/isaac-sim/IsaacLab/issues). For Isaac Sim specific issues, consult the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for questions, ideas, and feature requests.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports, documentation issues, new features, or updates with a definite scope and deliverable.

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the NVIDIA Omniverse Community by reaching out to OmniverseCommunity@nvidia.com. Engage with other developers and grow the community on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is released under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension and related scripts are licensed under [Apache 2.0](LICENSE-mimic).  License files for dependencies and assets are in the [`docs/licenses`](docs/licenses) directory.

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