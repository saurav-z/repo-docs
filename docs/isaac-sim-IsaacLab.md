![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab: Revolutionizing Robotics Research with GPU-Accelerated Simulation

**Isaac Lab is an open-source, GPU-accelerated framework that simplifies robotics research workflows like reinforcement learning, imitation learning, and motion planning, built upon NVIDIA Isaac Sim.**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Isaac Lab streamlines robotics research by providing fast, accurate physics and sensor simulation using the power of NVIDIA Isaac Sim. This enables researchers to iterate quickly, accelerating the development of intelligent robots.

## Key Features

*   **Extensive Robot Library:** Includes 16 pre-built robot models, including manipulators, quadrupeds, and humanoids.
*   **Ready-to-Train Environments:** Offers over 30 environments compatible with popular RL frameworks like RSL RL, SKRL, RL Games, and Stable Baselines, as well as support for multi-agent reinforcement learning.
*   **Advanced Physics Simulation:** Supports rigid bodies, articulated systems, and deformable objects.
*   **Comprehensive Sensor Suite:** Features RGB/depth/segmentation cameras, IMUs, contact sensors, and ray casters.
*   **GPU Acceleration:** Enables faster simulation and computation, crucial for iterative processes like RL.
*   **Flexible Deployment:** Runs locally or in the cloud for large-scale projects.

## Getting Started

Explore the [documentation](https://isaac-sim.github.io/IsaacLab) for comprehensive tutorials and guides:

*   [Installation Steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
*   [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
*   [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
*   [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

Isaac Lab requires specific versions of Isaac Sim. Here's a compatibility overview:

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

Note that the `feature/isaacsim_5_0` will contain active updates and may contain some breaking changes
until the official Isaac Lab 2.2 release.
It currently requires the [Isaac Sim 5.0 branch](https://github.com/isaac-sim/IsaacSim) available on GitHub built from source.
Please refer to the README in the `feature/isaacsim_5_0` branch for instructions for using Isaac Lab with Isaac Sim 5.0.
We are actively working on introducing backwards compatibility support for Isaac Sim 4.5 for this branch.

## Contributing

We welcome community contributions!  See the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) for details.

## Show & Tell: Share Your Projects!

Share your projects, tutorials, and learning content in the [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) section to inspire others and foster collaboration.

## Troubleshooting

Find solutions to common issues in the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues). For Isaac Sim-specific issues, consult its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) or [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

*   Use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussions, questions, and feature requests.
*   Report bugs, documentation issues, and feature requests through GitHub [Issues](https://github.com/isaac-sim/IsaacLab/issues).

## Connect with the NVIDIA Omniverse Community

Share your projects and collaborate with the NVIDIA Omniverse Community by contacting OmniverseCommunity@nvidia.com or joining the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse).

## License

Isaac Lab is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic). Dependencies' license files are in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab builds on the [Orbit](https://isaac-orbit.github.io/) framework. Please cite it in academic publications:

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

**[Back to Top](#isaac-lab-revolutionizing-robotics-research-with-gpu-accelerated-simulation)**
```

Key improvements and SEO considerations:

*   **Clear, concise title:**  "Isaac Lab: Revolutionizing Robotics Research with GPU-Accelerated Simulation" immediately communicates the project's purpose and benefit.  Uses target keywords.
*   **One-sentence hook:** A strong introductory sentence to grab attention.
*   **Keyword Optimization:**  Repeated use of key phrases like "robotics research," "GPU-accelerated," "simulation," "NVIDIA Isaac Sim," "reinforcement learning," etc.
*   **Structured Headings:**  Improved readability and SEO with clear headings and subheadings.
*   **Bulleted Lists:** Makes key features easily scannable for users.
*   **Internal Linking:** Added a "Back to Top" link at the end, helping with navigation.
*   **External Linking:**  Links to all relevant pages.
*   **Concise Language:** Removed unnecessary wordiness.
*   **Call to Action:**  Encourages users to explore the documentation and community features.
*   **Overall Readability:**  Well-organized content that's easy to digest.
*   **Stronger Focus on Benefits:** The revised summary emphasizes *what* users can achieve with Isaac Lab.
*   **Metadata Enhancement:** The use of image alt-text is good for SEO.
*   **Actionable Language:** Uses verbs to encourage users to engage with the content (e.g. "Explore," "Share," "Find").
*   **Explicit Mention of Target Audience:** While implicit, the language is geared towards robotics researchers and developers.
*   **Link to Original Repo:** Added.