[![Isaac Lab](docs/source/_static/isaaclab.jpg)](https://github.com/isaac-sim/IsaacLab)

# Isaac Lab: Accelerate Your Robotics Research with GPU-Powered Simulation

Isaac Lab is a powerful, open-source framework built on NVIDIA Isaac Sim, designed to streamline robotics research workflows like reinforcement learning and motion planning, offering a fast and accurate simulation environment.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

*   **Extensive Robot Models:** Explore a diverse collection of 16 commonly available robot models, including manipulators, quadrupeds, and humanoids.
*   **Ready-to-Train Environments:** Access over 30 pre-built environments compatible with popular reinforcement learning frameworks like RSL RL, SKRL, RL Games, and Stable Baselines, including support for multi-agent reinforcement learning.
*   **Advanced Physics Simulation:** Leverage robust physics simulation with rigid bodies, articulated systems, and deformable objects for realistic interactions.
*   **Realistic Sensor Simulation:** Utilize RTX-based cameras, LIDAR, and contact sensors for accurate and detailed environmental perception.
*   **GPU-Accelerated Performance:** Benefit from GPU acceleration for faster simulations and computations, critical for iterative processes like reinforcement learning.
*   **Flexible Deployment:** Run simulations locally or on the cloud for scalability and flexibility.

## Getting Started

Get started with Isaac Lab by following the steps below, and then dive into our in-depth documentation.

### 1. Prerequisites: Isaac Sim

Isaac Lab depends on NVIDIA Isaac Sim. Make sure you have Isaac Sim installed and configured according to the official [Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start).

### 2. Clone and Set Up Isaac Lab

1.  Clone the Isaac Lab repository:

    ```bash
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd isaaclab
    ```

2.  Set up a symbolic link to your Isaac Sim installation.  Replace the paths as needed:

    *   **Linux:**
        ```bash
        ln -s ../IsaacSim/_build/linux-x86_64/release _isaac_sim
        ```

    *   **Windows:**
        ```bash
        mklink /D _isaac_sim ..\IsaacSim\_build\windows-x86_64\release
        ```

3.  Install Isaac Lab using the provided script.

    *   **Linux:**
        ```bash
        ./isaaclab.sh -i
        ```

    *   **Windows:**
        ```bash
        isaaclab.bat -i
        ```

4.  [Optional] Set up a virtual Python environment (recommended for dependency management).

    *   **Linux:**
        ```bash
        source _isaac_sim/setup_conda_env.sh
        ```

    *   **Windows:**
        ```bash
        _isaac_sim\setup_python_env.bat
        ```

### 3. Train Your First Model

Run a sample training script to verify your setup:

*   **Linux:**
    ```bash
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
    ```
*   **Windows:**
    ```bash
    isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
    ```

### 4. Explore the Documentation

*   **Installation:** [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   **Reinforcement Learning:** [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   **Tutorials:** [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   **Available Environments:** [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab requires specific Isaac Sim versions.  Ensure compatibility:

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

We welcome community contributions! Review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details.

## Show & Tell: Share Your Projects

Share your projects and inspire others in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of our Discussions.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general questions and feature requests.
*   Use Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports and specific, actionable tasks.

## Connect with the NVIDIA Omniverse Community

Join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) and contact OmniverseCommunity@nvidia.com to share your work and collaborate.

## License

Isaac Lab is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its standalone scripts are released under [Apache 2.0](LICENSE-mimic).

## Acknowledgement

Isaac Lab builds upon the [Orbit](https://isaac-orbit.github.io/) framework.  Please cite it in your academic publications:

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