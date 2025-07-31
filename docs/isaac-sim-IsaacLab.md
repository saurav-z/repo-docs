# Isaac Lab: Accelerate Robotics Research with NVIDIA Isaac Sim

**Isaac Lab is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim, unifying robotics research workflows for efficient and accurate simulation.**  Explore the power of sim-to-real transfer with our robust and versatile platform.  Learn more and contribute at the [original repository](https://github.com/isaac-sim/IsaacLab).

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Key Features

*   **Rich Robot Models:** Access a diverse range of 16 pre-built robot models, including manipulators, quadrupeds, and humanoids.
*   **Extensive Environments:** Train with over 30 ready-to-use environments compatible with popular RL frameworks like RSL RL, SKRL, RL Games, and Stable Baselines, including support for multi-agent reinforcement learning.
*   **Advanced Physics Simulation:** Utilize robust physics engines for rigid bodies, articulated systems, and deformable objects.
*   **Realistic Sensor Simulation:** Leverage RTX-based cameras, LIDAR, IMU, and contact sensors for accurate data acquisition.

## Getting Started

Explore our comprehensive [documentation](https://isaac-sim.github.io/IsaacLab) with tutorials and guides:

*   [Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Compatibility

Isaac Lab requires specific Isaac Sim versions for compatibility.

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.1`                      | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

*Note: The `feature/isaacsim_5_0` branch requires [Isaac Sim 5.0 branch](https://github.com/isaac-sim/IsaacSim) and may contain breaking changes.  Refer to the branch's README for details.*

## Contributing

We welcome contributions to enhance Isaac Lab.  Review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for bug reports, feature requests, and code contributions.

## Showcase Your Work

Share your projects and tutorials in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section of the `Discussions` area.

## Troubleshooting

Find solutions in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For Isaac Sim issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for general questions.
*   Use GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues) for specific issues and feature requests.

## Connect with the NVIDIA Omniverse Community

Share your projects and resources by contacting the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com. Join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). Dependency licenses are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab is inspired by the [Orbit](https://isaac-orbit.github.io/) framework; cite it in academic publications:

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