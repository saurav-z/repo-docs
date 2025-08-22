![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with NVIDIA Isaac Sim

**Isaac Lab is a powerful, open-source framework that leverages NVIDIA Isaac Sim to streamline robotics research, enabling faster and more efficient development in reinforcement learning, imitation learning, and motion planning.** Explore the [Isaac Lab repository](https://github.com/isaac-sim/IsaacLab) to get started!

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

*   **GPU-Accelerated Simulation:** Experience fast and accurate physics and sensor simulation, ideal for iterative processes.
*   **Extensive Robot Models:** Access a diverse collection of 16+ pre-built robot models including manipulators, quadrupeds, and humanoids.
*   **Ready-to-Train Environments:** Utilize over 30 pre-configured environments compatible with popular reinforcement learning frameworks like RSL RL, SKRL, RL Games, and Stable Baselines. Supports multi-agent reinforcement learning.
*   **Advanced Sensor Simulation:** Simulate realistic sensors including RTX-based cameras, LIDAR, and contact sensors.
*   **Flexible Deployment:** Run simulations locally or scale them across the cloud.

## Getting Started

### Prerequisites

*   **NVIDIA Isaac Sim:**  Isaac Lab is built on top of Isaac Sim.  Refer to the Isaac Sim documentation for installation:  [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)

### Installation

1.  **Clone Isaac Sim (if you haven't already):**
    ```bash
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2.  **Build Isaac Sim:**
    ```bash
    cd IsaacSim
    ./build.sh  # For Linux
    # Or
    ./build.bat # For Windows
    ```

3.  **Clone Isaac Lab:**
    ```bash
    cd ..  # Go back to the parent directory
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

6.  **(Optional) Set up a Virtual Python Environment (e.g., Conda):**

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

*   **Comprehensive Documentation:**  Find detailed tutorials and step-by-step guides on the [documentation page](https://isaac-sim.github.io/IsaacLab).
*   **Installation:** Learn about local installation steps:  [Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   **Reinforcement Learning:** Explore reinforcement learning capabilities: [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   **Tutorials:**  Get started with tutorials:  [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   **Available Environments:**  Discover the available environments:  [Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure compatibility by referencing the correct Isaac Sim version for your Isaac Lab release.

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

Your contributions are welcome! Review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) to learn how to contribute code, report issues, or suggest features.

## Show & Tell: Share Your Robotics Projects

Share your projects and learning experiences in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions area. Inspire others and foster collaborations!

## Troubleshooting

Consult the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common solutions or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues) if needed. For Isaac Sim-specific issues, refer to its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   **Discussions:** Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for questions, ideas, and feature requests.
*   **Issues:** Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports, documentation issues, and well-defined feature work.

## Connect with the NVIDIA Omniverse Community

Share your projects with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com, or join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic).  Dependency licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

This project is based on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following in academic publications:

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