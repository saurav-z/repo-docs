# XLeRobot: Build Your Own Affordable Embodied AI Robot 🤖

**XLeRobot is an open-source project offering a low-cost, dual-arm mobile robot platform for embodied AI, enabling you to explore robotics and AI without breaking the bank.** Learn more and contribute at the [XLeRobot GitHub repository](https://github.com/Vector-Wangel/XLeRobot).

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

*   **Affordable**: Starting at just \$660, XLeRobot makes embodied AI accessible to a wider audience.
*   **Rapid Assembly**: Build your robot in under 4 hours, making it a perfect project for hobbyists and researchers alike.
*   **Open Source**: Built upon the shoulders of giants [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), [Bambot](https://github.com/timqian/bambot), empowering you to customize and extend its capabilities.
*   **Dual-Arm Mobile Platform**: Equipped with dual arms for manipulation tasks.
*   **Simulation & Real-World Control**: Includes both simulation and real-world control options, including keyboard, Xbox controller, and Switch joycon.
*   **Extensive Documentation**: Comprehensive documentation to guide you through assembly, software setup, and more.
*   **Community Support**: Join the Discord community for support and collaboration.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/17e31979-bd5e-4790-be70-566ea8bb181e" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/96ff4a3e-3402-47a2-bc6b-b45137ee3fdd" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/f6d52acc-bc8d-46f6-b3cd-8821f0306a7f" width="250"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/59086300-3e6f-4a3c-b5e0-db893eeabc0c" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/4ddbc0ff-ca42-4ad0-94c6-4e0f4047fd01" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/7abc890e-9c9c-4983-8b25-122573028de5" width="250"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/e74a602b-0146-49c4-953d-3fa3b038a7f7" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/d8090b15-97f3-4abc-98c8-208ae79894d5" width="250"/></td>
    <td><img src="https://github.com/user-attachments/assets/8b54adc3-d61b-42a0-8985-ea28f2e8f64c" width="250"/></td>
  </tr>
</table>

## Pricing & Cost Breakdown

> [!NOTE]
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price (Buy all the parts yourself) | US | EU | CN |
| --- | --- | --- | --- |
| **Basic** (use your laptop, single RGB head cam) | **~$660** | **~€680** | **~¥3999** |
| ↑ Stereo dual-eye RGB head cam | +$30 | +€30 | +¥199 |
| + RasberryPi | +$79 | +€79 | +¥399 |
| ↑ RealSense RGBD head cam | +$220 | +€230 | +¥1499 |

---

## What's New

*   **2025-08-30:** XLeRobot 0.3.0 Release with final outfit touch up and household chores showcase demos. Assembly kit ready for purchase soon, stay tuned!
*   **2025-07-30:** Control XLeRobot in real life with keyboard/Xbox controller/Switch joycon. All bluetooth, no wifi needed and zero latency.
    ![rea](https://github.com/user-attachments/assets/de8f50ad-a370-406c-97fb-fc01638d5624)
*   **2025-07-08:** Simulation with updated urdfs, control scripts (support Quest3 VR, keyboard, Xbox controller, switch joycon), support for new hardware and cameras, RL environment. Get started in 15 min.
    ![vr](https://github.com/user-attachments/assets/68b77bea-fdcf-4f42-9cf0-efcf1b188358)
*   **2025-07-01:** Documentation website out for more organized tutorials, demos and resources.
*   **2025-06-13:** XLeRobot 0.2.0 hardware setup, the 1st version fully capable for autonomous household tasks, starts from 660$.

---

## Getting Started

> [!NOTE]
> If you are new to programming, please familiarize yourself with basic Python, Ubuntu, and GitHub.

1.  **Buy Your Parts:** [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html)
2.  **3D Print:** [3D Printing Instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html)
3.  **Assemble:** [Assembly Instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html)
4.  **Software Setup:** [Software Instructions](https://xlerobot.readthedocs.io/en/latest/software/index.html)

---

## Contribute

**We welcome contributions to XLeRobot!** Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for details.

**Main Contributors:**

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   [Zhuoyi Lu](https://lzhuoyi.github.io/Zhuoyi_Lu.github.io/): RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

This project builds upon the work of [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).

## About Me

[Gaotian/Vector Wang](https://vector-wangel.github.io/)

I am a CS graduate student at Rice University [RobotPi Lab](https://robotpilab.github.io/), focusing on robust object manipulation. I built XLeRobot as a personal hobby to instantiate my research theory, also to provide a low-cost platform for people who are interested in robotics and embodied AI to work with.

[![Star History Chart](https://api.star-history.com/svg?repos=Vector-Wangel/XLeRobot&type=Timeline)](https://star-history.com/#Vector-Wangel/XLeRobot&Timeline)

---

## Citation

If you use XLeRobot, please cite it:

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

## ⚠️ Disclaimer ⚠️

> [!NOTE]
> You are responsible for any physical and mental damages that may occur from building, using, or developing on XLeRobot.