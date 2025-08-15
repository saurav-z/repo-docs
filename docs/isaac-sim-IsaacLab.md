![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab** is an open-source, GPU-accelerated framework built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html), designed to streamline robotics research workflows like reinforcement learning and sim-to-real transfer, allowing you to develop cutting-edge robotic solutions faster.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features:

*   **Robotics Environments:** Access a diverse collection of 30+ ready-to-train environments, including manipulators, quadrupeds, and humanoids, compatible with popular RL frameworks.
*   **Extensive Robot Models:** Work with 16+ commonly available robot models.
*   **Advanced Physics and Sensors:** Leverage rigid bodies, articulated systems, deformable objects, and realistic sensor simulations (RGB/depth cameras, IMU, contact sensors, etc.).
*   **GPU Acceleration:** Benefit from accelerated simulations for faster training and data processing.
*   **Flexible Deployment:** Run simulations locally or in the cloud for scalable research.

## Getting Started

### 1. Prerequisites:

*   **NVIDIA Isaac Sim:** Isaac Lab relies on NVIDIA Isaac Sim. Follow the [Isaac Sim Quick Start Guide](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) for installation.

### 2. Clone and Setup Isaac Lab:

1.  Clone Isaac Sim:
    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```
2.  Build Isaac Sim:
    ```bash
    cd IsaacSim
    ./build.sh  # For Linux
    # ./build.bat # For Windows
    ```
3.  Clone Isaac Lab:
    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```
4.  Set up symlink (Linux):
    ```bash
    ln -s ../IsaacSim/_build/linux-x86_64/release _isaac_sim
    ```
    (Windows):
    ```bash
    mklink /D _isaac_sim ..\IsaacSim\_build\windows-x86_64\release
    ```
5.  Install Isaac Lab (Linux):
    ```bash
    ./isaaclab.sh -i
    ```
    (Windows):
    ```bash
    isaaclab.bat -i
    ```
6.  [Optional] Set up a virtual python environment (e.g. for Conda) (Linux):
    ```bash
    source _isaac_sim/setup_conda_env.sh
    ```
    (Windows):
    ```bash
    _isaac_sim\setup_python_env.bat
    ```
7.  Train! (Linux):
    ```bash
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
    ```
    (Windows):
    ```bash
    isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
    ```

### 3. Explore the Documentation

*   [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure you use the correct Isaac Sim version with your Isaac Lab release:

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

Contributions are welcome! Review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Showcase Your Work!

Share your projects and tutorials in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section!

## Troubleshooting

Refer to the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues). For Isaac Sim issues, check the [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support & Community

*   Discuss ideas, ask questions, and request features in the [Discussions](https://github.com/isaac-sim/IsaacLab/discussions).
*   Report bugs and track feature work via [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with the NVIDIA Omniverse Community

Share your project with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com or join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic).

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in academic publications:

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

**[Back to Top](https://github.com/isaac-sim/IsaacLab)**