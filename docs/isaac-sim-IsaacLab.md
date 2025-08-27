![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is a powerful, open-source framework built on NVIDIA Isaac Sim, designed to streamline robotics research workflows for reinforcement learning, imitation learning, and motion planning. [Explore the original repository here](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab empowers researchers with a comprehensive suite of tools for robotics simulation and learning:

*   **Diverse Robot Models:** Access a wide range of robots, including manipulators, quadrupeds, and humanoids, with 16 commonly available models.
*   **Ready-to-Train Environments:** Utilize over 30 pre-built environments compatible with popular reinforcement learning frameworks like RSL RL, SKRL, RL Games, and Stable Baselines.  Support for multi-agent reinforcement learning is also included.
*   **Realistic Physics:** Leverage rigid bodies, articulated systems, and deformable objects for accurate simulation.
*   **Advanced Sensors:** Simulate a variety of sensors including RGB/depth/segmentation cameras, IMU, contact sensors, and ray casters.

## Why Choose Isaac Lab?

Built upon the robust foundation of NVIDIA Isaac Sim, Isaac Lab offers:

*   **GPU-Accelerated Performance:** Experience faster simulations and computations, crucial for iterative processes like reinforcement learning.
*   **Sim-to-Real Transfer:**  Achieve accurate sensor simulation (RTX-based cameras, LiDAR, contact sensors, etc.) for seamless sim-to-real robot learning.
*   **Flexibility and Scalability:** Run simulations locally or distribute them across the cloud for large-scale deployments.
*   **Open Source:** Benefit from the collaborative community and easily modify for your needs.

## Getting Started

Follow these steps to set up and run your first simulation:

### 1. Prerequisites: Isaac Sim Setup

Isaac Lab relies on NVIDIA Isaac Sim. Begin by installing Isaac Sim:

1.  **Clone Isaac Sim:**
    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```
2.  **Build Isaac Sim:**
    ```bash
    cd IsaacSim
    ./build.sh  # or build.bat on Windows
    ```

### 2. Isaac Lab Installation

1.  **Clone Isaac Lab:**
    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```
2.  **Set up Symlink (Linux):**
    ```bash
    ln -s ../IsaacSim/_build/linux-x86_64/release _isaac_sim
    ```
    **Set up Symlink (Windows):**
    ```bash
    mklink /D _isaac_sim ..\IsaacSim\_build\windows-x86_64\release
    ```
3.  **Install Isaac Lab (Linux):**
    ```bash
    ./isaaclab.sh -i
    ```
    **Install Isaac Lab (Windows):**
    ```bash
    isaaclab.bat -i
    ```
4.  **(Optional) Set up a virtual Python environment:**
    ```bash
    source _isaac_sim/setup_conda_env.sh # Linux
    # or
    _isaac_sim\setup_python_env.bat # Windows
    ```
5.  **Train a Model:**
    ```bash
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless  # Linux
    # or
    isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless  # Windows
    ```

### Documentation & Resources

*   **Detailed Documentation:** Access comprehensive tutorials and guides on the [documentation page](https://isaac-sim.github.io/IsaacLab).
    *   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
    *   [Reinforcement Learning Examples](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
    *   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
    *   [Environment Overview](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

### Isaac Sim Version Compatibility

Ensure compatibility by using the correct Isaac Sim version with your Isaac Lab release:

| Isaac Lab Version | Isaac Sim Version |
| ----------------- | ----------------- |
| `main` branch     | Isaac Sim 4.5 / 5.0 |
| `v2.2.0`          | Isaac Sim 4.5 / 5.0 |
| `v2.1.1`          | Isaac Sim 4.5       |
| `v2.1.0`          | Isaac Sim 4.5       |
| `v2.0.2`          | Isaac Sim 4.5       |
| `v2.0.1`          | Isaac Sim 4.5       |
| `v2.0.0`          | Isaac Sim 4.5       |

## Contributing

We encourage community contributions!  Review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn how to contribute code, report bugs, or suggest features.

## Show & Tell: Share Your Projects

Showcase your work and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions.

## Troubleshooting

*   Refer to the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for solutions to common issues.
*   For Isaac Sim-specific issues, consult the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).
*   Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.
*   Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## Support & Community

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for community discussions, questions, and feature requests.
*   Submit issues on [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports and specific tasks.

## Connect with the NVIDIA Omniverse Community

Share your projects and collaborate with the community:

*   Contact the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com
*   Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse)

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic). Dependency licenses are available in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is built upon the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in your publications:

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