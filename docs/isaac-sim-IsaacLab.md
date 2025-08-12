![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

**Isaac Lab is a cutting-edge, open-source framework built on NVIDIA Isaac Sim, designed to streamline robotics research workflows by providing fast, accurate, and GPU-accelerated simulations.** Explore the possibilities and contribute to the future of robotics with [Isaac Lab](https://github.com/isaac-sim/IsaacLab)!

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

Isaac Lab offers a comprehensive suite of features to facilitate your robotics research:

*   **Diverse Robot Models:** Access a wide range of robot models, including manipulators, quadrupeds, and humanoids, with 16 commonly available models.
*   **Rich Environments:** Train your robots with over 30 ready-to-use environments, compatible with popular reinforcement learning frameworks like RSL RL, SKRL, RL Games, and Stable Baselines, as well as multi-agent reinforcement learning.
*   **Advanced Physics Simulation:** Leverage accurate and fast physics simulation with rigid bodies, articulated systems, and deformable objects.
*   **Realistic Sensor Simulation:** Utilize a variety of sensors, including RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, and ray casters.
*   **GPU-Accelerated Performance:** Benefit from GPU acceleration for faster simulations and computations, ideal for iterative processes like reinforcement learning.
*   **Flexible Deployment:** Run simulations locally or distribute them across the cloud for large-scale deployments.

## Getting Started

### Prerequisites:

*   [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) - Ensure you have a compatible version installed. See the version dependency table below.

### Installation

Follow these steps to get Isaac Lab up and running:

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

7.  **Train!**

    *   **Linux:**

        ```bash
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
        ```

    *   **Windows:**

        ```bash
        isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
        ```

### Documentation & Resources

*   **Documentation:** Find detailed tutorials, installation guides, and more at our [documentation page](https://isaac-sim.github.io/IsaacLab).
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

We welcome contributions from the community! Please see our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details on how to contribute.

## Show & Tell: Share Your Work

We encourage you to showcase your projects and tutorials in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of our Discussions. Share your work to inspire others and foster collaboration within the community.

## Troubleshooting & Support

*   **Troubleshooting:** Consult the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common issues.
*   **Issues:** Report bugs and feature requests via [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues).
*   **Isaac Sim Issues:** For issues related to Isaac Sim, refer to the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or the [Isaac Sim forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).
*   **Support:** Use [GitHub Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general questions and discussions.

## Connect with the NVIDIA Omniverse Community

For opportunities to spotlight your work, reach out to the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com.

Join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect with other developers and grow the ecosystem.

## License

The Isaac Lab framework is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and standalone scripts are licensed under [Apache 2.0](LICENSE-mimic). Dependency and asset licenses are located in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is built upon the foundation of the [Orbit](https://isaac-orbit.github.io/) framework. If you use Isaac Lab in your academic work, please cite the following paper:

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