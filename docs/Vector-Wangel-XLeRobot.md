# XLeRobot: Build Your Own Affordable Household Robot 🤖

**Bring embodied AI to your home with XLeRobot, a versatile and budget-friendly dual-arm mobile robot, all for the price of a smartphone!**  This open-source project provides a comprehensive guide to building your own household robot, empowering you to explore the exciting world of robotics and AI.  [View the original repository on GitHub](https://github.com/Vector-Wangel/XLeRobot).

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-brown.svg)](README_CN.md)

[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter/X](https://img.shields.io/twitter/follow/VectorWang?style=social)](https://twitter.com/VectorWang2)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://xlerobot.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/badge/Discord-XLeRobot-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/bjZveEUh6F)

---

## Key Features

*   **Affordable:** Starting at just $660 (USD), XLeRobot is designed to be budget-friendly.
*   **Fast Assembly:** Build your robot in under 4 hours!
*   **Open Source:** Built upon the giants: [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), [Bambot](https://github.com/timqian/bambot)
*   **Versatile Control:** Control XLeRobot using a keyboard, Xbox controller, or Switch joycon.
*   **Simulation Support:** Test and develop your robot in a simulated environment.
*   **Comprehensive Documentation:**  Detailed tutorials and resources available on the [XLeRobot Documentation Website](https://xlerobot.readthedocs.io/en/latest/index.html).

---

## Cost Breakdown

> [!NOTE]
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price                                       | US       | EU       | CN       |
| ------------------------------------------- | -------- | -------- | -------- |
| **Basic** (use your laptop, single RGB head cam) | **~$660** | **~€680** | **~¥3999** |
| ↑ Stereo dual-eye RGB head cam             | +$30     | +€30     | +¥199    |
| + RasberryPi                               | +$79     | +€79     | +¥399    |
| ↑ RealSense RGBD head cam                  | +$220    | +€230    | +¥1499   |

---

## Recent Updates

*   **2025-07-30:** [Control XLeRobot in real life](https://xlerobot.readthedocs.io/en/latest/software/index.html) with keyboard/Xbox controller/Switch joycon, all bluetooth, no wifi needed and zero latency.
    ![rea](https://github.com/user-attachments/assets/de8f50ad-a370-406c-97fb-fc01638d5624)

*   **2025-07-08:** [**Simulation**](https://xlerobot.readthedocs.io/en/latest/simulation/index.html) with updated urdfs, control scripts (support Quest3 VR, keyboard, Xbox controller, switch joycon), support for new hardware and cameras, RL environment.
    ![vr](https://github.com/user-attachments/assets/68b77bea-fdcf-4f42-9cf0-efcf1b188358)

*   **2025-07-01:** [**Documentation** website](https://xlerobot.readthedocs.io/en/latest/index.html) out for more orgainized turotials, demos and resources.

*   **2025-06-13:** [**XLeRobot 0.2.0**](https://xlerobot.readthedocs.io) hardware setup, the 1st version fully capable for autonomous household tasks, starts from 660$.

---

## Getting Started

> [!NOTE]
> This guide is designed for beginners.

> [!NOTE]
> Familiarity with basic Python, Ubuntu, and Github is recommended.

1.  💵 **Buy your parts**: [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html)
2.  🖨️ **Print your stuff**: [3D printing](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html)
3.  🔨 **Assemble**: [Assemble!](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html)
4.  💻 **Software**: [Get your robot moving!](https://xlerobot.readthedocs.io/en/latest/software/index.html)

---

## Core Contributors

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   Zhuoyi Lu: RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

This project is built upon the work of: [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).

We welcome contributions to the project!

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Vector-Wangel/XLeRobot&type=Timeline)](https://star-history.com/#Vector-Wangel/XLeRobot&Timeline)

---

## Citation

```bibtex
@misc{wang2025xlerobot,
    author = {Wang, Gaotian}, {Lu, Zhuoyi}
    title = {XLeRobot: A Practical Low-cost Household Dual-Arm Mobile Robot Design for General Manipulation},
    howpublished = "\url{https://github.com/Vector-Wangel/XLeRobot}",
    year = {2025}
}
```

---

## ⚠️ Disclaimer ⚠️

> [!NOTE]
> You are fully responsible for any physical or mental damages resulting from building, buying, or developing a XLeRobot based on this repository.