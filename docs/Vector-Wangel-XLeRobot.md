# XLeRobot: Build Your Own Affordable Embodied AI Robot 🤖

[XLeRobot](https://github.com/Vector-Wangel/XLeRobot) offers a cost-effective platform for anyone interested in embodied AI, enabling you to build a dual-arm mobile robot for less than the price of a new iPhone!

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

*   **Affordable**: Starting at just $660, XLeRobot makes embodied AI accessible to everyone.
*   **Fast Assembly**: Build your robot in under 4 hours.
*   **Dual-Arm Mobile Robot**:  Provides advanced manipulation capabilities for various tasks.
*   **Open Source & Customizable**:  Built on open-source foundations, allowing for extensive modification and experimentation.
*   **Simulation and Real-World Control**:  Includes simulation environments for easy testing and control options like keyboard, Xbox controller, and Joycons, with zero latency!
*   **Active Development**:  Regular updates with new features and demos, including real-world household chore capabilities.

## What's New
*   **Developer Assembly kit (excluding battery and IKEA cart) ready for purchase**: China (Taobao) and world-wide (Wowrobo), details in the original README.
*   **Embodied AI Home Robot Hackathon Mentor**: XLeRobot is a mentor at the SEEED x Nvidia x Huggingface Hackathon.
*   **XLeRobot 0.3.0 Release**: Including the final touch up and household chores showcases.
*   **Real-life Control**: With keyboard/Xbox controller/Switch joycon. All bluetooth, no wifi needed.
*   **Simulation**: Get started in 15 min with updated urdfs, control scripts, support for new hardware and cameras, and RL environment.
*   **Detailed Documentation**: Organized tutorials, demos and resources on the documentation website.
*   **XLeRobot 0.2.0**: Hardware setup and first version fully capable for autonomous household tasks.

---

## Cost Breakdown

> [!NOTE]
> Costs exclude 3D printing, tools, shipping, and taxes.

| Price (Buy all the parts yourself) | US         | EU         | CN        | IN         |
| :------------------------------- | :---------- | :---------- | :--------- | :---------- |
| **Basic** (use your laptop)    | **~$660**    | **~€680**    | **~¥3999**  | **~₹87000**  |
| ↑ Stereo dual-eye RGB head cam  | +$30       | +€30       | +¥199      | +₹6550      |
| + RasberryPi                     | +$79       | +€79       | +¥399      | +₹7999      |
| ↑ RealSense RGBD head cam       | +$220      | +€230      | +¥1499     | +₹35726     |

---

## Get Started

> [!NOTE]
> If you are totally new to programming, please spend at least a day to get yourself familiar with basic Python, Ubuntu and Github (with the help of Google and AI). At least you should know how to setup ubuntu system, git clone, pip install, use intepreters (VS Code, Cursor, Pycharm, etc.) and directly run commands in the terminals.

1.  💵 **Buy your parts**: [Bill of Materials](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/material.html)
2.  🖨️ **Print your stuff**: [3D printing](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/3d.html)
3.  🔨 **Assemble**! [Assemble instructions](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html)
4.  💻 **Software**: [Get your robot moving!](https://xlerobot.readthedocs.io/en/latest/software/index.html)

---

## Contribute

**Interested in helping out?**  Check out the [CONTRIBUTING.md](CONTRIBUTING.md) file for how to get involved!

**Core Contributors:**

*   [Gaotian/Vector Wang](https://vector-wangel.github.io/)
*   [Zhuoyi Lu](https://lzhuoyi.github.io/Zhuoyi_Lu.github.io/): RL sim2real deploy, teleop on real robot (Xbox, VR, Joycon)
*   Yiyang Huang: RL & VLA implementation (ongoing)
*   YCP: WebUI for remote control (ongoing)
*   [Lixing Zhang](lixingzhang.com): Hardware design improvements
*   Nicole Yue: Documentation website setup
*   Yuesong Wang: Mujoco simulation

XLeRobot is built upon the shoulders of giants, including [LeRobot](https://github.com/huggingface/lerobot), [SO-100](https://github.com/TheRobotStudio/SO-ARM100), [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi), and [Bambot](https://github.com/timqian/bambot).

---

## About the Author

[Gaotian/Vector Wang](https://vector-wangel.github.io/), a CS graduate student at Rice University's RobotPi Lab, built XLeRobot to explore robust object manipulation and make robotics accessible.

[![Star History Chart](https://api.star-history.com/svg?repos=Vector-Wangel/XLeRobot&type=Timeline)](https://star-history.com/#Vector-Wangel/XLeRobot&Timeline)

---

## Citation

If you use this work, please cite it:

```bibtex
@misc{wang2025xlerobot,
    author = {Wang, Gaotian and Lu, Zhuoyi},
    title = {XLeRobot: A Practical Low-cost Household Dual-Arm Mobile Robot Design for General Manipulation},
    howpublished = "\url{https://github.com/Vector-Wangel/XLeRobot}",
    year = {2025}
}
```
---
## 🪧 Disclaimer 🪧

> [!NOTE]
> You are responsible for any damages from building or developing using this repo.