# Isaac Lab: Accelerate Robotics Research with GPU-Powered Simulation

**Isaac Lab is an open-source, GPU-accelerated framework built on NVIDIA Isaac Sim that simplifies robotics research, enabling faster and more accurate simulation for reinforcement learning, imitation learning, and motion planning.**

[![Isaac Lab](docs/source/_static/isaaclab.jpg)](https://github.com/isaac-sim/IsaacLab)

## Key Features:

*   **Accelerated Physics and Simulation:** Harness the power of GPUs for rapid and precise robotics simulation.
*   **Rich Robotic Environments:** Access a diverse library of robots, environments, and sensors to jumpstart your research.
*   **Sim-to-Real Transfer:**  Leverage accurate sensor simulation, including RTX-based cameras and LIDAR, to bridge the gap between simulation and real-world applications.
*   **Scalable Deployment:** Run simulations locally or scale them across the cloud for large-scale projects.
*   **Extensive Environment Support:** Train models within a wide variety of environments and with popular RL frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines.

## Get Started

Explore our comprehensive [documentation](https://isaac-sim.github.io/IsaacLab) for detailed tutorials, installation guides, and environment overviews to kickstart your robotics projects.

*   [Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Compatibility

Isaac Lab is designed to work with specific versions of NVIDIA Isaac Sim.  Refer to the table below for version compatibility:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

Note: The `feature/isaacsim_5_0` branch is under active development and may contain breaking changes.

## Contribute

We encourage community contributions!  Please review our [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) and share your ideas through [Discussions](https://github.com/isaac-sim/IsaacLab/discussions).

## Troubleshooting and Support

Find solutions to common issues in our [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or report problems on [GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues).  For Isaac Sim-specific questions, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67). For general inquiries, use the [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) section.

## Share Your Work

Showcase your projects, tutorials, and learning content in our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section!

## License

Isaac Lab is licensed under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic).

## Acknowledgement

Isaac Lab is built on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite the following in academic publications:

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

**[Visit the Isaac Lab Repository on GitHub](https://github.com/isaac-sim/IsaacLab)**