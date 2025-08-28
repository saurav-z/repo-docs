# XLeRobot: Affordable & Accessible Embodied AI for Everyone

**Build your own dual-arm mobile robot for under $660 and unlock the future of robotics today!** Learn more and contribute at the [original XLeRobot repository](https://github.com/Vector-Wangel/XLeRobot).

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-brown.svg)](README_CN.md)

<a href="https://xlerobot.readthedocs.io/en/latest/index.html">
  <img width="1725" height="1140" alt="front" src="https://github.com/user-attachments/assets/f9c454ee-2c46-42b4-a5d7-88834a1c95ab" />
</a>

[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter/X](https://img.shields.io/twitter/follow/VectorWang?style=social)](https://twitter.com/VectorWang2)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://xlerobot.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/badge/Discord-XLeRobot-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/bjZveEUh6F)

---

## Key Features

*   **Cost-Effective:** Build a fully functional dual-arm mobile robot for as little as $660 (excluding 3D printing, tools, shipping, and taxes).
*   **Fast Assembly:**  Assemble your XLeRobot in less than 4 hours.
*   **Open Source & Extensible:** Built upon the foundations of [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot), allowing for easy modification and customization.
*   **Multiple Control Options:** Control XLeRobot using a keyboard, Xbox controller, or Switch joycon (Bluetooth, no WiFi needed, zero latency).
*   **Simulation Support:**  Develop and test your robot in a realistic simulation environment, with updated URDFs and control scripts.
*   **Comprehensive Documentation:**  Get started quickly with detailed tutorials, demos, and resources available on the [documentation website](https://xlerobot.readthedocs.io/en/latest/index.html).
*   **Active Development:**  Stay up-to-date with the latest features and improvements.
*   **Community Driven:** Contribute to the project and collaborate with other robotics enthusiasts.

## Cost Breakdown

> [!NOTE]
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price                                          | US      | EU      | CN      |
| :--------------------------------------------- | :------ | :------ | :------ |
| **Basic** (use your laptop, single RGB head cam) | **~$660** | **~€680** | **~¥3999** |
| ↑ Stereo dual-eye RGB head cam                 | +$30    | +€30    | +¥199   |
| + RasberryPi                                  | +$79    | +€79    | +¥399   |
| ↑ RealSense RGBD head cam                     | +$220   | +€230   | +¥1499  |

---

## Recent Updates & News

*   **2025-07-30:** Control XLeRobot in real life with keyboard, Xbox controller, or Switch joycon. All bluetooth, no wifi needed and zero latency.
    ![rea](https://github.com/user-attachments/assets/de8f50ad-a370-406c-97fb-fc01638d5624)

*   **2025-07-08:** Simulation environment with updated URDFs, control scripts (support Quest3 VR, keyboard, Xbox controller, switch joycon), support for new hardware and cameras, and RL environment.
    ![vr](https://github.com/user-attachments/assets/68b77bea-fdcf-4f42-9cf0-efcf1b188358)

*   **2025-07-01:** Documentation website launched for organized tutorials, demos, and resources.
*   **2025-06-13:** XLeRobot 0.2.0 hardware setup released, the 1st version fully capable for autonomous household tasks, starts from 660$.

---

## Getting Started

> [!NOTE]
> If you are totally new to programming, please spend at least a day to get yourself familiar with basic Python, Ubuntu and Github (with the help of Google and AI). At least you should know how to set up ubuntu system, git clone, pip install, use intepreters (VS Code, Cursor, Pycharm, etc.) and directly run commands in the terminals.

1.  💵 **Buy your parts**: [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html)
2.  🖨️ **Print your stuff**: [3D printing](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html)
3.  🔨 **Assemble**! [Assembly Instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html)
4.  💻 **Software**: [Get your robot moving!](https://xlerobot.readthedocs.io/en/latest/software/index.html)

---

## Contributing

**🤝  Want to contribute to XLeRobot?**  See the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidance on how to get involved!

**Main Contributors**

*   Zhuoyi Lu: RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

---

## About the Creator

**Gaotian/Vector Wang** - [https://vector-wangel.github.io/](https://vector-wangel.github.io/)

A CS graduate student at Rice University, focused on robust object manipulation, where we propse virtual cages and funnels and physics-aware world models to close the Sim2real gap and achieve robust manipulation under uncertainties. One of my papers, Caging in Time, has recently been accepted by International Journal of Robotics Research (IJRR).

I built XLeRobot as a personal hobby to instantiate my research theory, also to provide a low-cost platform for people who are interested in robotics and embodied AI to work with.

[![Star History Chart](https://api.star-history.com/svg?repos=Vector-Wangel/XLeRobot&type=Timeline)](https://star-history.com/#Vector-Wangel/XLeRobot&Timeline)

---

## Citation

If you want, you can cite this work with:

```bibtex
@misc{wang2025xlerobot,
    author = {Wang, Gaotian and Lu, Zhuoyi},
    title = {XLeRobot: A Practical Low-cost Household Dual-Arm Mobile Robot Design for General Manipulation},
    howpublished = "\url{https://github.com/Vector-Wangel/XLeRobot}",
    year = {2025}
}
```

---
![Generated Image August 27, 2025 - 4_58PM](https://github.com/user-attachments/assets/682ef049-bb42-4b50-bf98-74d6311e774d)

---

## ⚠️ Disclaimer ⚠️

> [!NOTE]
> If you build, buy, or develop a XLeRobot based on this repo, you will be fully responsible for all the physical and mental damages it does to you or others.