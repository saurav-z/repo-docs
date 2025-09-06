![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Unleash Robotics Research with GPU-Accelerated Simulation

**Accelerate your robotics research with Isaac Lab, an open-source framework built on NVIDIA Isaac Sim for realistic, GPU-accelerated simulations.** Learn more and contribute at the [original IsaacLab repo](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

*   **Robots:** Explore 16+ pre-configured robot models, including manipulators, quadrupeds, and humanoids.
*   **Environments:** Access over 30 ready-to-train environments compatible with popular reinforcement learning frameworks (RSL RL, SKRL, RL Games, Stable Baselines, and multi-agent RL).
*   **Physics:** Utilize advanced physics simulation with rigid bodies, articulated systems, and deformable objects.
*   **Sensors:** Benefit from accurate sensor simulation, including RGB/depth/segmentation cameras, IMUs, and contact sensors.

## Getting Started

### Prerequisites

*   NVIDIA Isaac Sim: This framework is built on top of Isaac Sim. Make sure you have access to the correct version:
    *   `main` branch: Isaac Sim 4.5 / 5.0
    *   `v2.2.X`: Isaac Sim 4.5 / 5.0
    *   `v2.1.X`: Isaac Sim 4.5
    *   `v2.0.X`: Isaac Sim 4.5

### Installation

1.  **Clone Isaac Sim:**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh # Linux
    # or
    build.bat # Windows
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

6.  **(Optional) Set up a virtual Python environment (e.g., for Conda):**

    *   **Linux:**

        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```

    *   **Windows:**

        ```bash
        _isaac_sim\setup_python_env.bat
        ```

7.  **Train Your Robot!**

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation

*   **Installation:** [Installation Steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   **Reinforcement Learning:** [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   **Tutorials:** [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   **Environments:** [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

| Isaac Lab Version | Isaac Sim Version |
| :---------------- | :---------------- |
| `main` branch     | Isaac Sim 4.5 / 5.0 |
| `v2.2.X`          | Isaac Sim 4.5 / 5.0 |
| `v2.1.X`          | Isaac Sim 4.5       |
| `v2.0.X`          | Isaac Sim 4.5       |

## Contribute

We welcome contributions from the community! Review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to get started.

## Show & Tell

Share your projects and learning experiences in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions.

## Troubleshooting

*   Check the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common issues.
*   Report issues on [GitHub](https://github.com/isaac-sim/IsaacLab/issues).
*   For Isaac Sim issues, consult the [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussions, questions, and feature requests.
*   Use [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports, documentation issues, and specific feature implementations.

## Connect with the NVIDIA Omniverse Community

Share your work and collaborate with others via [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) or by contacting the NVIDIA Omniverse Community team: OmniverseCommunity@nvidia.com.

## License

*   [BSD-3 License](LICENSE)
*   `isaaclab_mimic` extension: [Apache 2.0](LICENSE-mimic)
*   Dependencies and assets: [`docs/licenses`](docs/licenses)

## Acknowledgement

Isaac Lab is based on the [Orbit](https://isaac-orbit.github.io/) framework.  Cite it in academic publications:

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