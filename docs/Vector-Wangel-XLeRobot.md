# XLeRobot: Affordable & Accessible Embodied AI for Everyone

**XLeRobot is a cutting-edge, open-source dual-arm mobile robot designed to bring the power of embodied AI to your home, all while being cheaper than an iPhone!** Explore the original project on [GitHub](https://github.com/Vector-Wangel/XLeRobot).

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-brown.svg)](README_CN.md)

[Link to Documentation Website](https://xlerobot.readthedocs.io/en/latest/index.html)

[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter/X](https://img.shields.io/twitter/follow/VectorWang?style=social)](https://twitter.com/VectorWang2)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://xlerobot.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/badge/Discord-XLeRobot-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/bjZveEUh6F)
---

## Key Features & Benefits

*   **Affordable:** Starting at just ~$660, making advanced robotics accessible.
*   **Quick Assembly:** Build your XLeRobot in under 4 hours!
*   **Dual-Arm Capabilities:** Designed for general manipulation tasks in a household setting.
*   **Open-Source:** Built upon the shoulders of giants: LeRobot, SO-100/SO-101, Lekiwi, and Bambot - explore, modify, and contribute!
*   **Versatile Control:** Control XLeRobot in real-life using keyboard/Xbox controller/Switch joycon over bluetooth, zero latency.
*   **Simulation Support:** Comprehensive simulation environment with updated URDFs and control scripts.
*   **Detailed Documentation:** Extensive documentation and tutorials to help you get started quickly.

## Cost Breakdown

> [!NOTE]
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price | US | EU | CN |
| --- | --- | --- | --- |
| **Basic** (use your laptop, single RGB head cam) | **~$660** | **~€680** | **~¥3999** |
| ↑ Stereo dual-eye RGB head cam | +$30 | +€30 | +¥199 |
| + RasberryPi | +$79 | +€79 | +¥399 |
| ↑ RealSense RGBD head cam | +$220 | +€230 | +¥1499 |

---

## What's New

*   **2025-07-30:** Control XLeRobot in real life with keyboard/Xbox controller/Switch joycon. All bluetooth, no wifi needed and zero latency.
    ![rea](https://github.com/user-attachments/assets/de8f50ad-a370-406c-97fb-fc01638d5624)

*   **2025-07-08:** Simulation with updated urdfs, control scripts (support Quest3 VR, keyboard, Xbox controller, switch joycon), support for new hardware and cameras, RL environment. Get started in 15 min.
    ![vr](https://github.com/user-attachments/assets/68b77bea-fdcf-4f42-9cf0-efcf1b188358)

*   **2025-07-01:** Documentation website out for more orgainized turotials, demos and resources.

*   **2025-06-13:** XLeRobot 0.2.0 hardware setup, the 1st version fully capable for autonomous household tasks, starts from 660$.

---

## Getting Started

> [!NOTE]
> This guide is designed to be friendly for beginners. Familiarity with Python, Ubuntu, and Git is recommended.

1.  💵 **Buy your parts**: [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html)
2.  🖨️ **Print your stuff**: [3D printing](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html)
3.  🔨 **Assemble**! [Assembly Guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html)
4.  💻 **Software**: [Software Setup](https://xlerobot.readthedocs.io/en/latest/software/index.html)

---

## Main Contributors

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   Zhuoyi Lu: RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

This project is a collaborative effort, made possible by the work of numerous individuals and projects, including [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).

We welcome contributions from anyone interested in advancing this exciting project!

[![Star History Chart](https://api.star-history.com/svg?repos=Vector-Wangel/XLeRobot&type=Timeline)](https://star-history.com/#Vector-Wangel/XLeRobot&Timeline)

---

## Citation

If you use this work, please cite it as:

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
> You are responsible for any damages that may occur from building or using the XLeRobot.