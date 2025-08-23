![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Accelerate Robotics Research with NVIDIA Isaac Sim

**Isaac Lab empowers robotics researchers with a GPU-accelerated, open-source framework built on NVIDIA Isaac Sim, streamlining workflows for reinforcement learning, imitation learning, and motion planning.**  For more details, visit the [original repository](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab is a cutting-edge framework designed to unify and simplify robotics research, providing an efficient platform for projects involving reinforcement learning, imitation learning, and motion planning.  Built on NVIDIA Isaac Sim, it leverages fast and accurate physics and sensor simulation for optimal sim-to-real transfer.

## Key Features:

*   **Simulated Robots:** Access a diverse selection of 16 robot models, including manipulators, quadrupeds, and humanoids.
*   **Pre-built Environments:** Train with over 30 ready-to-use environments compatible with popular reinforcement learning frameworks like RSL RL, SKRL, and Stable Baselines, including support for multi-agent reinforcement learning.
*   **Advanced Physics Simulation:** Utilize rigid bodies, articulated systems, and deformable objects for realistic robot behavior.
*   **Comprehensive Sensor Suite:** Integrate RGB/depth/segmentation cameras, IMU, contact sensors, and ray casters for accurate environmental perception.
*   **GPU-Accelerated Performance:** Experience faster simulation and computation, crucial for iterative processes like reinforcement learning.
*   **Flexible Deployment:** Run simulations locally or distribute them across the cloud for large-scale experiments.

## Getting Started:

### Prerequisites

*   [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) (Required)

### Installation

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

6.  **[Optional] Set up a virtual python environment (e.g. for Conda):**

    ```bash
    source _isaac_sim/setup_conda_env.sh  # Linux
    # or
    _isaac_sim\setup_python_env.bat     # Windows
    ```

7.  **Run Training (Linux):**

    ```bash
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
    ```

    **Run Training (Windows):**

    ```bash
    isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
    ```

### Documentation:

Explore comprehensive documentation for detailed guidance:

*   [Installation Instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Ensure compatibility with the following Isaac Sim versions for each Isaac Lab release:

| Isaac Lab Version | Isaac Sim Version |
|-------------------|-------------------|
| `main`            | 4.5 / 5.0         |
| `v2.2.0`          | 4.5 / 5.0         |
| `v2.1.1`          | 4.5               |
| `v2.1.0`          | 4.5               |
| `v2.0.2`          | 4.5               |
| `v2.0.1`          | 4.5               |
| `v2.0.0`          | 4.5               |

## Contributing

Contribute to Isaac Lab's development through bug reports, feature requests, and code contributions. Review the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details.

## Showcase Your Projects

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the Discussions. Inspire others and foster collaboration within the community!

## Troubleshooting

Refer to the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for common fixes or submit an issue [here](https://github.com/isaac-sim/IsaacLab/issues).

For Isaac Sim-related issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or the [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Utilize GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for inquiries and feature requests.
*   Submit [Issues](https://github.com/isaac-sim/IsaacLab/issues) for bug reports, documentation problems, and defined work items.

## Connect with the NVIDIA Omniverse Community

Share your projects by reaching out to the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com.
Join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to engage with other developers.

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE).  The `isaaclab_mimic` extension uses the [Apache 2.0](LICENSE-mimic) license. Dependency and asset licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgements

This project builds upon the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following paper if you use Isaac Lab in your research:

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