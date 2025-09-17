# XLeRobot: Affordable Dual-Arm Mobile Robot for Everyone

**XLeRobot offers an accessible and cost-effective entry point into the world of embodied AI, allowing you to build a capable dual-arm mobile robot for less than the price of a new smartphone!**  [Explore the project on GitHub](https://github.com/Vector-Wangel/XLeRobot)

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

*   **Low-Cost:**  Starting at just \$660 USD, XLeRobot is designed to be budget-friendly.
*   **Fast Assembly:** Assemble your robot in under 4 hours.
*   **Dual-Arm Capabilities:**  Offers advanced manipulation for household tasks.
*   **Open Source:**  Built upon open-source projects and encourages community contributions.
*   **Comprehensive Documentation:**  Detailed documentation and tutorials at [xlerobot.readthedocs.io](https://xlerobot.readthedocs.io/en/latest/).
*   **Multiple Control Options:** Control via keyboard, Xbox controller, or Switch joycon.
*   **Simulation Support:**  Get started quickly with pre-built simulations.

---

## What's New

**Latest Updates:**

*   **[2025-09-09]** **Developer Assembly Kit Available:**  Purchase a developer kit (excluding battery and IKEA cart) in [China (Taobao) for 3699￥](https://e.tb.cn/h.SZFbBgZABZ8zRPe?tk=ba514rTBRjQ) and [worldwide for \$579](https://shop.wowrobo.com/products/xlerobot-dual-arm-mobile-household-robot-kit?variant=47297659961561).
    *   This kit, a collaboration with Wowrobo, is designed to provide an accessible entry point for developers.
    *   The price is kept low.
    *   Check the documentation website and this repo for codes and tutorials.
*   **[2025-09-09]**  XLeRobot will be mentored by **SEEED x Nvidia x Huggingface** in the [Embodied AI Home Robot Hackathon](https://www.seeedstudio.com/embodied-ai-worldwide-hackathon-home-robot.html) (Oct 25–26, Bay Area)! [Register HERE](https://docs.google.com/forms/d/e/1FAIpQLSdYYDegdgIypxuGJNLcoc8kbdmU4jKgl49zg4X-107LAmBN4g/viewform).

<img width="2400" height="1256" alt="image" src="https://github.com/user-attachments/assets/4132c23b-5c86-4bb9-94b4-a6b12059685b" />
*   **[2025-08-30]**  XLeRobot 0.3.0 Release with final outfit touch up and household chores showcase demos.
*   **[2025-07-30]**  Real-life robot control with keyboard/Xbox controller/Switch joycon.
    ![rea](https://github.com/user-attachments/assets/de8f50ad-a370-406c-97fb-fc01638d5624)
*   **[2025-07-08]**  Simulation with updated urdfs, control scripts, and RL environment.
    ![vr](https://github.com/user-attachments/assets/68b77bea-fdcf-4f42-9cf0-efcf1b188358)
*   **[2025-07-01]**  Comprehensive [Documentation Website](https://xlerobot.readthedocs.io/en/latest/) launched.
*   **[2025-06-13]**  XLeRobot 0.2.0 Release:  Hardware setup for autonomous household tasks.

---

## Cost Breakdown

> [!NOTE]
> Costs exclude 3D printing, tools, shipping, and taxes.

| Price (Buy all the parts yourself) | US        | EU       | CN       | IN       |
| :---------------------------------- | :-------- | :------- | :------- | :------- |
| **Basic** (single RGB head cam)     | **~$660** | **~€680** | **~¥3999** | **~₹87000** |
| ↑ Stereo dual-eye RGB head cam      | +$30     | +€30     | +¥199    | +₹6550   |
| + RaspberryPi                       | +$79     | +€79     | +¥399    | +₹7999   |
| ↑ RealSense RGBD head cam           | +$220    | +€230    | +¥1499   | +₹35726  |

---

## Getting Started

Follow these simple steps to begin your XLeRobot journey:

1.  **Buy Parts:** [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html)
2.  **3D Print:** [3D Printing Instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html)
3.  **Assemble:** [Assembly Guide](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html)
4.  **Software:** [Robot Software Setup](https://xlerobot.readthedocs.io/en/latest/software/index.html)

> [!NOTE]
> New to programming?  Familiarize yourself with Python, Ubuntu, and Git before you start.  Learn the basics of setup, cloning, and running commands in your terminal.

---

## Contribute

**We welcome contributions!**  Please review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Core Contributors:**

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   [Zhuoyi Lu](https://lzhuoyi.github.io/Zhuoyi_Lu.github.io/): RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Yiyang Huang: RL & VLA implementation (ongoing)
*   YCP: WebUI for remote control (ongoing)
*   [Lixing Zhang](lixingzhang.com): Hardware design improvements
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

**Acknowledgments:**
XLeRobot builds upon the work of [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).  We appreciate the contributions of all contributors.

---

## About the Creator

[Gaotian/Vector Wang](https://vector-wangel.github.io/) is a CS graduate student at Rice University's RobotPi Lab. He is focused on object manipulation and built XLeRobot to explore his research and provide an accessible platform for embodied AI and robotics enthusiasts.

[![Star History Chart](https://api.star-history.com/svg?repos=Vector-Wangel/XLeRobot&type=Timeline)](https://star-history.com/#Vector-Wangel/XLeRobot&Timeline)

---

## Citation

```bibtex
@misc{wang2025xlerobot,
    author = {Wang, Gaotian and Lu, Zhuoyi},
    title = {XLeRobot: A Practical Low-cost Household Dual-Arm Mobile Robot Design for General Manipulation},
    howpublished = "\url{https://github.com/Vector-Wangel/XLeRobot}",
    year = {2025}
}
```
---
## Disclaimer

> [!NOTE]
> You are fully responsible for any damages that may occur from building, purchasing, or developing XLeRobot.