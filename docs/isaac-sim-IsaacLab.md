![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Isaac Lab is a GPU-accelerated, open-source framework that simplifies robotics research by leveraging the power of NVIDIA Isaac Sim for fast and accurate simulation.**  Discover how you can build, train, and test advanced robotics solutions in a virtual environment.

[Explore the original Isaac Lab repository](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

*   **Accelerated Simulation:** Leverage GPU acceleration for faster simulation and computation, crucial for iterative processes like reinforcement learning.
*   **Extensive Robot Models:** Includes a diverse collection of 16+ pre-built robot models, from manipulators to humanoids.
*   **Rich Environments:** Access to 30+ ready-to-train environments for reinforcement learning, imitation learning, and motion planning. Supports popular RL frameworks such as RSL RL, SKRL, RL Games, and Stable Baselines, as well as multi-agent reinforcement learning.
*   **Advanced Physics and Sensors:** Accurate simulation of rigid bodies, articulated systems, deformable objects, and realistic sensor data including RGB/depth/segmentation cameras, IMUs, and contact sensors.
*   **Sim-to-Real Focus:** Built on NVIDIA Isaac Sim, designed to facilitate sim-to-real transfer.
*   **Flexible Deployment:** Run simulations locally or in the cloud for large-scale experimentation.

## Getting Started

### Prerequisites

*   [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) (Installation instructions in Isaac Sim README)
*   Python 3.11

### Installation

1.  **Clone Isaac Sim:**

    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**

    ```bash
    cd IsaacSim
    ./build.sh  # Linux
    # or
    ./build.bat # Windows
    ```

3.  **Clone Isaac Lab:**

    ```bash
    cd ..
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

4.  **Set up symlink:**

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

6.  **(Optional) Set up a virtual Python environment:**

    *   **Linux:**

        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```

    *   **Windows:**

        ```bash
        _isaac_sim\setup_python_env.bat
        ```

7.  **Train a model:**

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

## Documentation

Access comprehensive documentation to guide your journey with Isaac Lab:

*   [Installation Steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning Guide](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure compatibility by referencing the following table for Isaac Lab and Isaac Sim version pairings:

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

We welcome community contributions!  Review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn how to contribute bug reports, feature requests, and code.

## Show & Tell: Share Your Projects

The [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of our Discussions area is a place to showcase your projects and learning experiences, inspiring others and fostering collaboration.

## Troubleshooting

Consult the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues). For issues with Isaac Sim, refer to the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general discussions and questions.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) to report bugs, documentation problems, and track new features.

## Connect with the NVIDIA Omniverse Community

To spotlight your work, contact the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com.  Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to collaborate and build with other developers.

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its scripts are under [Apache 2.0](LICENSE-mimic).  Dependencies and asset licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab builds upon the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following in your publications:

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