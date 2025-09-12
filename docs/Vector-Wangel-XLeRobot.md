# XLeRobot: Affordable Dual-Arm Mobile Robot for Embodied AI

**Build your own household robot for under $700 with XLeRobot – bringing embodied AI within reach!** Explore the project on [GitHub](https://github.com/Vector-Wangel/XLeRobot).

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

*   **Low Cost:** Starting at $660, making advanced robotics accessible.
*   **Fast Assembly:** Build your robot in under 4 hours.
*   **Dual-Arm Design:** Enables complex manipulation tasks.
*   **Open Source:** Built upon established projects like LeRobot, SO-100/SO-101, Lekiwi, and Bambot, promoting collaboration.
*   **Multiple Control Options:** Control via keyboard, Xbox controller, or Switch Joycon.
*   **Simulation Support:** Includes simulation environments for development and testing.
*   **Active Development:** Regular updates with new features and improvements.

---

## What's New

*   **Developer Assembly Kit Available:** A ready-to-purchase kit (excluding battery and IKEA cart) is available in China and worldwide. See details below:
    *   **China (Taobao):** 3699￥ [Link](https://e.tb.cn/h.SZFbBgZABZ8zRPe?tk=ba514rTBRjQ)
    *   **Worldwide:** $579 [Link](https://shop.wowrobo.com/products/xlerobot-dual-arm-mobile-household-robot-kit?variant=47297659961561)
    <img width="1482" height="485" alt="image" src="https://github.com/user-attachments/assets/788836c1-966a-4d11-a911-5c37befc0b85" />
*   **Embodied AI Home Robot Hackathon Mentor:** XLeRobot is participating in the SEEED x Nvidia x Huggingface Embodied AI Home Robot Hackathon (October 25–26, Bay Area). [Register Here](https://docs.google.com/forms/d/e/1FAIpQLSdYYDegdgIypxuGJNLcoc8kbdmU4jKgl49zg4X-107LAmBN4g/viewform).
*   **XLeRobot 0.3.0 Release:** Final outfit touch-up and household chore demo.
*   **Real-time Control:** Control XLeRobot remotely with zero latency using keyboard, Xbox controller, or Switch Joycon.
    ![rea](https://github.com/user-attachments/assets/de8f50ad-a370-406c-97fb-fc01638d5624)
*   **Simulation:** Updated URDFs, control scripts (support Quest3 VR, keyboard, Xbox controller, switch joycon), and RL environment.
    ![vr](https://github.com/user-attachments/assets/68b77ea-fdcf-4f42-9cf0-efcf1b188358)
*   **Comprehensive Documentation:** Tutorials, demos, and resources are available on the [documentation website](https://xlerobot.readthedocs.io/en/latest/index.html).
*   **XLeRobot 0.2.0 Release:** Hardware setup and autonomous household task capabilities.

---

## Cost Breakdown

> [!NOTE]
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price (Buy all the parts yourself) | US | EU | CN |
| --- | --- | --- | --- |
| **Basic** (use your laptop, single RGB head cam) | **~$660** | **~€680** | **~¥3999** |
| ↑ Stereo dual-eye RGB head cam | +$30 | +€30 | +¥199 |
| + RasberryPi | +$79 | +€79 | +¥399 |
| ↑ RealSense RGBD head cam | +$220 | +€230 | +¥1499 |

---

## Getting Started

> [!NOTE]
> If you're new to programming, familiarize yourself with Python, Ubuntu, and Github before proceeding.

1.  **Purchase Parts:** Refer to the [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html).
2.  **3D Printing:** Follow the [3D printing](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html) guide.
3.  **Assemble:** [Assemble](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html) the robot.
4.  **Software:** Get your robot moving with the [software instructions](https://xlerobot.readthedocs.io/en/latest/software/index.html).

---

## Contribute

**Interested in contributing?** See the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

**Key Contributors:**

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   [Zhuoyi Lu](https://lzhuoyi.github.io/Zhuoyi_Lu.github.io/): RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Yiyang Huang: RL & VLA implementation (ongoing)
*   YCP: WebUI for remote control (ongoing)
*   [Lixing Zhang](lixingzhang.com): Hardware design improvements
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

**Acknowledgments:**

XLeRobot builds upon the work of [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).

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

## About the Developer

[Gaotian/Vector Wang](https://vector-wangel.github.io/) is a CS graduate student at Rice University's RobotPi Lab, researching robust object manipulation and bridging the sim-to-real gap. XLeRobot is a personal project aimed at making robotics accessible and a platform for research.

[![Star History Chart](https://api.star-history.com/svg?repos=Vector-Wangel/XLeRobot&type=Timeline)](https://star-history.com/#Vector-Wangel/XLeRobot&Timeline)

---

## 🪧 Disclaimer 🪧

> [!NOTE]
> You are responsible for any damages caused by your XLeRobot build.