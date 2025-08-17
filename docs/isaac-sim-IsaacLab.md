![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab is an open-source framework built on NVIDIA Isaac Sim that empowers robotics researchers to streamline their workflows and achieve sim-to-real transfer.** Explore the full potential of robotics simulation and training with our comprehensive tools and environments. [Learn more and contribute on GitHub](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab offers a comprehensive suite of tools and environments to accelerate your robotics research:

*   **Extensive Robot Models:** Access a diverse range of robots, including manipulators, quadrupeds, and humanoids, with 16 commonly available models.
*   **Rich Environment Support:** Train your models in over 30 ready-to-use environments compatible with leading reinforcement learning frameworks like RSL RL, SKRL, RL Games, and Stable Baselines. Multi-agent reinforcement learning is also supported.
*   **Realistic Physics:** Benefit from accurate simulations of rigid bodies, articulated systems, and deformable objects.
*   **Advanced Sensor Simulation:** Utilize RTX-based cameras, LIDAR, IMU, contact sensors, and ray casters for realistic data acquisition.

## Getting Started

### Prerequisites

*   NVIDIA Isaac Sim (Refer to [Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) for installation)
*   Python 3.11

### Installation

1.  **Clone Isaac Sim (If you haven't already):**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh  # Or build.bat on Windows
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

6.  **(Optional) Set up a Virtual Environment (e.g., Conda):**

    *   **Linux:**

        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```
    *   **Windows:**

        ```bash
        _isaac_sim\setup_python_env.bat
        ```

7.  **Train Your First Model!**

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```
    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation & Resources

*   **Detailed Documentation:** Explore our comprehensive [documentation page](https://isaac-sim.github.io/IsaacLab) for in-depth tutorials and guides.
*   **Installation Guide:**  [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   **Reinforcement Learning:** [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   **Tutorials:** [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   **Environments:** [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

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

We welcome contributions from the community! Review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn how to contribute.

## Show & Tell: Share Your Projects

Share your creations and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of our Discussions.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section.  For Isaac Sim specific issues, consult the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67). Submit issues [here](https://github.com/isaac-sim/IsaacLab/issues).

## Support & Community

*   **Discussions:** Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for questions and feature requests.
*   **Issues:** Report bugs, documentation problems, or feature requests via GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with the NVIDIA Omniverse Community

Share your projects and resources with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com. Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) for collaborative development and ecosystem growth.

## License

Isaac Lab is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). Dependencies and assets are licensed within the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is based on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following in academic publications:

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