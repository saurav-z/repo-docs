![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab**, built upon NVIDIA Isaac Sim, is a GPU-accelerated, open-source framework designed to streamline robotics research, offering rapid and accurate physics and sensor simulation for tasks like reinforcement learning, imitation learning, and motion planning. **Explore the original repository on GitHub: [Isaac Lab](https://github.com/isaac-sim/IsaacLab)**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab provides a comprehensive toolkit for robotics simulation and learning:

*   **Diverse Robot Models:** Includes a wide selection of 16 commonly available robot models, including manipulators, quadrupeds, and humanoids.
*   **Extensive Environment Library:** Offers over 30 ready-to-train environments, compatible with popular reinforcement learning frameworks. Supports single and multi-agent reinforcement learning.
*   **Advanced Physics Engine:** Integrates a robust physics engine for simulating rigid bodies, articulated systems, and deformable objects.
*   **Realistic Sensor Simulation:** Features accurate sensor models, including RGB/depth/segmentation cameras, IMUs, contact sensors, and ray casters.

## Getting Started

### Prerequisites:
*   [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html). See the [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start) for installation.
*   Python 3.11

### Installation Steps
1.  Clone Isaac Sim
    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  Build Isaac Sim
    ```bash
    cd IsaacSim
    ./build.sh  # For Linux
    # or use build.bat for Windows
    ```

3.  Clone Isaac Lab
    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

4.  Set up symlink
    ```bash
    # Linux:
    ln -s ../IsaacSim/_build/linux-x86_64/release _isaac_sim
    # Windows:
    mklink /D _isaac_sim ..\IsaacSim\_build\windows-x86_64\release
    ```

5.  Install Isaac Lab
    ```bash
    # Linux:
    ./isaaclab.sh -i
    # Windows:
    isaaclab.bat -i
    ```

6.  [Optional] Set up a virtual python environment (e.g. for Conda)
    ```bash
    # Linux:
    source _isaac_sim/setup_conda_env.sh
    # Windows:
    _isaac_sim\setup_python_env.bat
    ```

7.  Train!
    ```bash
    # Linux:
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
    # Windows:
    isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
    ```

### Documentation

Comprehensive documentation is available to help you get started:

*   [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Examples](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

Ensure compatibility by referencing the version table:

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

Your contributions are highly valued! Please refer to the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details.

## Show & Tell

Share your projects and tutorials in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the repository's Discussions to inspire the community.

## Troubleshooting

*   Refer to the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common solutions.
*   Report issues via [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues).
*   For Isaac Sim-specific issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussions, questions, and feature requests.
*   Use [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports and actionable tasks.

## Connect with the NVIDIA Omniverse Community

Share your work with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com or joining the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is released under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension is under [Apache 2.0](LICENSE-mimic). License files for dependencies are located in [`docs/licenses`](docs/licenses).

## Acknowledgement

Isaac Lab is based on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following paper if you use Isaac Lab in your research:

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