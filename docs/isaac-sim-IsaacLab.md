# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

**Isaac Lab is an open-source, GPU-accelerated robotics simulation framework built on NVIDIA Isaac Sim, designed to streamline robotics research and enable efficient sim-to-real transfer.**  This powerful framework provides a comprehensive suite of tools for researchers and developers working on reinforcement learning, imitation learning, and motion planning.

[**Explore the Isaac Lab Repository on GitHub**](https://github.com/isaac-sim/IsaacLab)

## Key Features

*   **Diverse Robot Models:** Includes 16+ pre-built robot models, including manipulators, quadrupeds, and humanoids, ready for simulation and experimentation.
*   **Extensive Environments:** Offers over 30 pre-configured environments, compatible with popular RL frameworks like RSL RL, SKRL, RL Games, and Stable Baselines, with support for multi-agent RL.
*   **Realistic Physics:**  Simulates rigid bodies, articulated systems, and deformable objects with high fidelity.
*   **Advanced Sensor Simulation:** Provides accurate simulation of various sensors, including RGB/depth/segmentation cameras, IMUs, contact sensors, and ray casters.
*   **GPU-Accelerated Performance:** Leverages the power of GPUs for faster simulations and computations, accelerating iterative processes like reinforcement learning and data-intensive tasks.
*   **Flexible Deployment:** Runs locally or in the cloud, enabling scalability for large-scale experiments.

## Getting Started

### 1. Prerequisites: NVIDIA Isaac Sim

Isaac Lab is built on NVIDIA Isaac Sim.  Follow the official instructions for installing Isaac Sim.

*   **Quick Start:**  [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start).

### 2. Installation Steps

1.  **Clone Isaac Sim:**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh  # For Linux
    # or
    .\build.bat # For Windows
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

6.  **[Optional] Set up a Python Environment:**

    *   **Linux (Conda Example):**

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

### Documentation & Tutorials

Dive deeper into Isaac Lab with comprehensive documentation and tutorials:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Examples](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure compatibility by using the correct Isaac Sim version with your Isaac Lab release:

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

Join our community and help improve Isaac Lab!  We welcome contributions in the form of bug reports, feature requests, and code contributions.  See our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details.

## Show & Tell

Share your projects and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of our Discussions.

## Troubleshooting

Find solutions to common issues and report problems:

*   [Troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html)
*   [Submit an Issue](https://github.com/isaac-sim/IsaacLab/issues)

For Isaac Sim-specific issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussion and feature requests.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) to track specific, actionable work items (bugs, documentation, features).

## Connect with the NVIDIA Omniverse Community

Share your work and collaborate with the community! Contact the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com or join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic). Dependency licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab builds upon the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in your academic publications:

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