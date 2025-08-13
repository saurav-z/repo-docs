![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab is an open-source framework built on NVIDIA Isaac Sim, empowering robotics researchers with GPU-accelerated simulation for reinforcement learning, imitation learning, and motion planning.**  Explore the power of realistic physics and sensor simulation for your robotics projects.  Find out more on the original repository [here](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

*   **Realistic Simulation:** Leverage the power of NVIDIA Isaac Sim for accurate physics and sensor simulation, enabling sim-to-real transfer.
*   **GPU Acceleration:**  Run complex simulations and computations faster with GPU acceleration, crucial for iterative processes like reinforcement learning.
*   **Diverse Robotics Support:**  Includes a comprehensive selection of robots, environments, physics and sensors for a wide variety of research applications.

    *   **Robots:** 16+ pre-built robot models (manipulators, quadrupeds, humanoids).
    *   **Environments:**  30+ ready-to-train environments, compatible with popular RL frameworks (RSL RL, SKRL, RL Games, Stable Baselines) and MARL.
    *   **Physics:** Support for rigid bodies, articulated systems, and deformable objects.
    *   **Sensors:**  Simulated RGB/depth/segmentation cameras, IMUs, contact sensors, and ray casters.
*   **Flexible Deployment:** Run simulations locally or in the cloud for scalability.

## Getting Started

### Prerequisites

Before you begin, ensure you have installed NVIDIA Isaac Sim. Detailed installation instructions are in the [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start).

1.  **Clone Isaac Sim:**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh  # or build.bat on Windows
    ```

3.  **Clone Isaac Lab:**

    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

4.  **Set up Symlink (Linux):**

    ```bash
    ln -s ../IsaacSim/_build/linux-x86_64/release _isaac_sim
    ```

    **Set up Symlink (Windows):**

    ```bash
    mklink /D _isaac_sim ..\IsaacSim\_build\windows-x86_64\release
    ```

5.  **Install Isaac Lab (Linux):**

    ```bash
    ./isaaclab.sh -i
    ```

    **Install Isaac Lab (Windows):**

    ```bash
    isaaclab.bat -i
    ```

6.  **(Optional) Set up a Virtual Environment:**

    **Linux (Conda):**

    ```bash
    source _isaac_sim/setup_conda_env.sh
    ```

    **Windows:**

    ```bash
    _isaac_sim\setup_python_env.bat
    ```

7.  **Train!**

    **Linux:**

    ```bash
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
    ```

    **Windows:**

    ```bash
    isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
    ```

### Documentation

Find comprehensive guides and tutorials at our documentation page:

*   [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure you use the correct Isaac Sim version for your Isaac Lab release:

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

We welcome community contributions!  Review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details.

## Show & Tell: Share Your Projects

Showcase your work and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section.

## Troubleshooting

Consult the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) guide for common issues or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For Isaac Sim-specific problems, check the [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions and feature requests.
*   Report bugs and track development in GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with the NVIDIA Omniverse Community

Share your projects and collaborate by contacting the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com or joining the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is under [Apache 2.0](LICENSE-mimic). Dependencies are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

This project is based on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following paper:

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